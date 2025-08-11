import numpy as np
import itertools
from scipy import stats, special
from sklearn.linear_model import LinearRegression

# --- Sampling Functions ---

def get_permutations(n_samples, players, n_variables):
    perms = np.random.binomial(n=1, p=0.5, size=(n_samples, len(players)))
    var_matrix = np.zeros((n_samples, n_variables))
    for i, player in enumerate(players):
        for member in player:
            var_matrix[:, member] = perms[:, i]
    return var_matrix

def sample_marginal(players, instance, dataset, n_samples):
    n_variables = len(instance)
    permutation = get_permutations(n_samples, players, n_variables)
    modified_data = np.repeat(instance[np.newaxis, :], n_samples, axis=0)
    mask = permutation.astype(bool)
    background_samples = dataset[np.random.randint(0, dataset.shape[0], size=n_samples)]
    modified_data[mask] = background_samples[mask]
    coalition = np.logical_not(permutation)
    return modified_data, coalition

def get_coalition_samples(coalition, instance, dataset, sampling_method, n_samples):
    outside_coalition = [i for i in range(len(instance)) if i not in coalition]
    modified_data = np.repeat(instance[None, :], n_samples, axis=0)
    if not outside_coalition:
        return modified_data
    
    if sampling_method == "marginal":
        indices = np.random.randint(0, dataset.shape[0], size=n_samples)
        for i in outside_coalition:
            modified_data[:, i] = dataset[indices, i]
    elif sampling_method == "independent":
        for i in outside_coalition:
            modified_data[:, i] = dataset[np.random.randint(0, dataset.shape[0], size=n_samples), i]
    else:
        raise ValueError("Sampling method must be 'marginal' or 'independent'")
    return modified_data

def sample_data(players, instance, dataset, sampling_method, n_samples):
    if sampling_method == "marginal":
        return sample_marginal(players, instance, dataset, n_samples)
    # Add independent sampling if needed
    raise ValueError("Only 'marginal' sampling is fully implemented in this refactor.")

# --- Partitioning and Math Functions ---

def test_additivity(predictions, coalition, i, j, alpha):
    """Performs a t-test to check for non-additive effects between features i and j."""
    i_active = coalition[:, i]
    j_active = coalition[:, j]
    
    sample_none = predictions[np.logical_and(~i_active, ~j_active)]
    sample_both = predictions[np.logical_and(i_active, j_active)]
    sample_i = predictions[np.logical_and(i_active, ~j_active)]
    sample_j = predictions[np.logical_and(~i_active, j_active)]
    
    # Ensure there are enough samples for the test
    if any(len(s) < 2 for s in [sample_none, sample_both, sample_i, sample_j]):
        return False # Cannot test, assume additivity

    sample_i_test_1 = sample_i - np.random.choice(sample_none, len(sample_i))
    sample_i_test_2 = np.random.choice(sample_both, len(sample_j)) - sample_j
    sample_j_test_1 = sample_j - np.random.choice(sample_none, len(sample_j))
    sample_j_test_2 = np.random.choice(sample_both, len(sample_i)) - sample_i

    _, p_i = stats.ttest_ind(sample_i_test_1, sample_i_test_2, equal_var=False)
    _, p_j = stats.ttest_ind(sample_j_test_1, sample_j_test_2, equal_var=False)

    return p_i < (alpha / 2) or p_j < (alpha / 2)

def get_partition_recursive(collection):
    if len(collection) == 1:
        yield [collection]
        return
    first = collection[0]
    for smaller in get_partition_recursive(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n+1:]
        yield [[first]] + smaller

def generate_max_partition(max_partition):
    combs = [get_partition_recursive(s) for s in max_partition]
    for comb in itertools.product(*combs):
        yield sum(comb, [])


def lasso_term(partition, lambd):
    c = np.array([len(x) * (len(x) - 1) / 2 for x in partition])
    return lambd * np.sum(c)

def binomial_weight_kernel(players, coalition):
    player_matrix = coalition[:, [p[0] for p in players]]
    active_players = np.sum(player_matrix, axis=1)
    d = len(players)
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = (d - 1) / (special.binom(d, active_players) * active_players * (d - active_players))
    weights[np.isinf(weights) | np.isnan(weights)] = 0
    return weights

def compute_coalition_shapleys(players, model, instance, dataset, is_class, samp_method, n_samples):
    if len(players) == 1:
        pred_func = model.predict_proba if is_class else model.predict
        value = pred_func(instance.reshape(1,-1))[0]
        if is_class: value = value[1]
        return [(players[0], value)]
        
    data, coalition = sample_data(players, instance, dataset, samp_method, n_samples)
    predictions = model.predict_proba(data)[:, 1] if is_class else model.predict(data)
    
    player_indices = [p[0] for p in players]
    player_matrix = coalition[:, player_indices]
    
    weights = binomial_weight_kernel(players, coalition)
    
    lm = LinearRegression()
    lm.fit(player_matrix, predictions, sample_weight=weights)
    
    return list(zip(players, lm.coef_))

def compute_value_function(coalition, model, instance, average_pred, dataset, is_classification, sampling_method, n_samples):
    """
    Computes the value function for a given coalition of features.
    
    This is the expected model output when the coalition's features are known (fixed to the
    instance's values) and the other features are unknown (sampled from the background data).
    """
    data = get_coalition_samples(coalition, instance, dataset, sampling_method, n_samples)
    
    if is_classification:
        predictions = model.predict_proba(data)[:, 1]
    else:
        predictions = model.predict(data)
        
    return np.mean(predictions) - average_pred
