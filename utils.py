import numpy as np
import itertools

def get_permutations(n_samples, players, n_variables):
    perms = np.random.binomial(n=1,p=0.5,size=(n_samples,len(players)))
    var_matrix = np.zeros((n_samples,n_variables))
    for i in range(len(players)):
        player = players[i]
        for member in player:
            var_matrix[:,member] = perms[:,i]
    return var_matrix

def compute_value_function(coalition,model,instance,average_pred, dataset,is_classification,sampling_method,n_samples):
    data = get_coalition_samples(coalition,instance, dataset, sampling_method, n_samples)
    predictions = model.predict(data) if not is_classification else model.predict_proba(data)[:,1]
    value_function =  np.mean(predictions)    
    return value_function - average_pred

def sample_marginal(players,instance, dataset, n_samples):
    n_variables = len(instance)
    permutation = get_permutations(n_samples, players, n_variables)

    modified_data = np.repeat(instance[np.newaxis,:], n_samples, axis=0)

    mask = np.ma.make_mask(permutation)
    modified_data[mask] = dataset[np.random.randint(0,dataset.shape[0],size=(n_samples,)),:][mask]
    
    coalition = np.logical_not(permutation)
    return modified_data, coalition

def sample_independent(players,instance, dataset, n_samples):
    n_variables = len(instance)
    permutation = get_permutations(n_samples, players, n_variables)

    modified_data = np.repeat(instance[np.newaxis,:], n_samples, axis=0)
    mask = np.ma.make_mask(permutation)
    for i in range(n_variables):
        submask = mask[:,i]
        modified_data[submask,i] = dataset[np.random.randint(0,dataset.shape[0],size=(n_samples,)),i][submask]
    
    coalition = np.logical_not(permutation)
    return modified_data, coalition

def get_coalition_samples(coalition,instance, dataset, sampling_method, n_samples):
    outside_coalition = [i for i in range(len(instance)) if i not in coalition]
    modified_data = np.repeat(instance[None,:], n_samples, axis=0)
    if len(outside_coalition) == 0:
        return modified_data
    
    if sampling_method == "marginal":
        indices = np.random.randint(0,dataset.shape[0],size=(n_samples,))
        for i in outside_coalition:
            modified_data[:,i] = dataset[indices,i]
        
    elif sampling_method == "independent":
        for i in outside_coalition:
            modified_data[:,i] = dataset[np.random.randint(0,dataset.shape[0],size=(n_samples,)),i]
    else:
        raise ValueError("Sampling method must be either 'marginal' or 'independent'")
    return modified_data

def sample_data(players,instance, dataset, sampling_method, n_samples, *args):
    if sampling_method == "independent":
        return sample_independent(players,instance, dataset, n_samples)
    elif sampling_method == "marginal":
        return sample_marginal(players,instance,dataset,n_samples)
    else:
        raise ValueError("Sampling method must be either 'marginal' or 'independent'")
    
def next_merge(P):
    newPs = set()
    for i in P: # does i,j and j,i since newPs is a set will only contian once
        for j in P:
            if i == j:
                continue
            newP = i.union(j)
            set_P = set(P)
            set_P.discard(i)
            set_P.discard(j)
            set_P.add(newP)
            newPs.add(frozenset(set_P))
    return newPs

# check if valid finer partition of max_partition
def valid_sub_partition(P, max_partition):
    valid_subset = True
    for s in P: 
        subset_found = False
        for sM in max_partition:
            if s.issubset(sM):
                subset_found = True
                break
        if not subset_found:
            valid_subset = False
            break
    return valid_subset

def get_all_partitions(d, max_partition=None):
    # create max_partition or check type
    if max_partition is None:
        max_partition = frozenset({frozenset([i for i in range(d)])})
    else:
        assert isinstance(max_partition, frozenset)
        for s in max_partition:
            assert isinstance(s, frozenset)

    all_partitions = set()
    P = frozenset([frozenset([i]) for i in range(d)])
    all_partitions.add(P)
    open_p = next_merge(P)
    while len(open_p) > 0:
        next_open = set()
        for p in open_p:
            if valid_sub_partition(p, max_partition):
                all_partitions.add(p)
                next_p = next_merge(p)
                next_open = next_open.union(next_p)
        open_p = next_open
    return all_partitions


def get_partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in get_partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller

def generate_max_partition(max_partition):
    combs = []
    for s in max_partition:
        combs.append(get_partition(s))
    partitions = []
    for comb in itertools.product(*combs):
        partitions.append(sum(comb, []))
    return partitions
