import numpy as np
from utils import *
import networkx as nx
from scipy import stats, special
import copy
from sklearn.linear_model import LinearRegression
from networkx.algorithms.connectivity.edge_kcomponents import bridge_components, k_edge_components
import pickle


'''
Code for Interaction Aware Explanations
Based on the paper: Interaction Aware Explanations
The algorithm consists of three main stages:
1. Determine possible pairings through tests of non-additive effects and if possible split graph into subgraphs through connected components
2. Within each subgraph, find the partition which minimizes the disparity between the sum of effects and the effect of the partition
3. With the full partition, compute the final explanation for each element, which is either the value function or SHAP
'''

def find_interactions(model,average_prediction,instance,dataset,is_classification,sampling_method,n_samples,alpha):
    players = [(i,) for i in range(dataset.shape[1])]
    data, coalition = sample_data(players,instance, dataset, sampling_method, n_samples)    
    predictions = model.predict(data) if not is_classification else model.predict_proba(data)[:,1]

    interaction_graph = nx.Graph()
    interaction_graph.add_nodes_from(list(range(len(players))))
    for i in range(len(players)):
        for j in range(i+1,len(players)):
            rejected = test_additivity(predictions,coalition, i, j,alpha)
            if rejected:
                interaction_graph.add_edge(i,j)
    return interaction_graph

def test_additivity(predictions, coalition, i, j, alpha):
    i_active = coalition[:,i]
    j_active = coalition[:,j]
    both_active = np.logical_and(i_active,j_active)
    none_active = np.logical_and(np.logical_not(i_active),np.logical_not(j_active))
    only_i_active = np.logical_and(i_active,np.logical_not(j_active))
    only_j_active = np.logical_and(j_active,np.logical_not(i_active))

    sample_none = predictions[none_active]
    sample_both = predictions[both_active]
    sample_i = predictions[only_i_active]
    sample_j = predictions[only_j_active]

    sample_i_test_1 = sample_i - sample_none[np.random.choice(none_active.sum(),only_i_active.sum())]
    sample_i_test_2 = sample_both[np.random.choice(both_active.sum(),only_j_active.sum()) ] - sample_j

    sample_j_test_1 = sample_j - sample_none[np.random.choice(none_active.sum(),only_j_active.sum())]
    sample_j_test_2 = sample_both[np.random.choice(both_active.sum(),only_i_active.sum()) ] - sample_i

    _,p_i = stats.ttest_ind(sample_i_test_1,sample_i_test_2,equal_var=False)
    _,p_j = stats.ttest_ind(sample_j_test_1,sample_j_test_2,equal_var=False)

    adjusted_alpha = alpha/2
    return p_i < adjusted_alpha or p_j < adjusted_alpha

    

def is_subclique(G,nodelist):
    H = G.subgraph(nodelist)
    n = len(nodelist)
    return H.size() == n*(n-1)/2


def find_partition_exhaustive(interactions_graph,average_prediction,model,instance,dataset,is_classification,sampling_method,n_samples,max_coalition_size,lambd):

    d = len(instance)
    max_partition = [list(range(d))]
    if interactions_graph is not None:
        components = nx.connected_components(interactions_graph)
        max_partition = []
        for c in components:
            max_partition.append(list(c))

    value_target = model.predict(instance.reshape(1,-1))-average_prediction if not is_classification else model.predict_proba(instance.reshape(1,-1))[:,1] - average_prediction
    value_function_buffer = {}
    component_sum = 0
    value_steps = 0
    for coalition in [(i,) for i in range(d)]:
        coalition_sample = get_coalition_samples(coalition,instance, dataset, sampling_method, n_samples)
        predictions = model.predict(coalition_sample) if not is_classification else model.predict_proba(coalition_sample)[:,1]
        value_function = np.mean(predictions)- average_prediction
        value_function_buffer[coalition] = value_function
        component_sum += value_function
        value_steps += 1
    lambd = lambd*(component_sum-value_target)**2

    if interactions_graph is None:
        all_partitions = generate_max_partition(max_partition)
    else:
        all_partitions = generate_max_partition(max_partition)
    top_partition = None
    top_score = 0
    eligible_partitions = 0
    for partition in all_partitions:

        partition = list(map(lambda x: tuple(x),partition))
        
        if any(map(lambda x: len(x) > max_coalition_size,partition)):# or any(map(lambda x: not is_subclique(interactions_graph,x),partition)):
            continue
        eligible_partitions += 1

        partition_value = 0
        for coalition in partition:
            coalition_value = 0
            if coalition in value_function_buffer:
                coalition_value = value_function_buffer[coalition]
            else:
                coalition_sample = get_coalition_samples(coalition,instance, dataset, sampling_method, n_samples)
                coalition_value = np.mean(model.predict(coalition_sample))-average_prediction if not is_classification else np.mean(model.predict_proba(coalition_sample)[:,1])- average_prediction
                value_function_buffer[coalition] = coalition_value
                value_steps += 1
            partition_value += coalition_value
        objective = (partition_value - value_target)**2 + lasso_term(partition,lambd)
        if objective < top_score or top_partition is None:
            top_score = objective
            top_partition = partition
    return top_partition, eligible_partitions, value_steps


def find_partition_greedy(interactions_graph,average_prediction,model,instance,dataset,is_classification,sampling_method,n_samples,max_coalition_size,lambd):
    partition = [(i,) for i in range(dataset.shape[1])]
    value_target = 0
    value_target = model.predict(instance.reshape(1,-1))-average_prediction if not is_classification else model.predict_proba(instance.reshape(1,-1))[:,1] - average_prediction
    # compute the value function of each singleton
    component_sum = 0
    partition_values = {}
    for coalition in partition:
        coalition_sample = get_coalition_samples(coalition,instance, dataset, sampling_method, n_samples)
        predictions = model.predict(coalition_sample) if not is_classification else model.predict_proba(coalition_sample)[:,1]
        value_function = np.mean(predictions)- average_prediction
        partition_values[coalition] = value_function
        component_sum += value_function
    lambd = lambd*(component_sum-value_target)**2
    regularization_term = lasso_term(partition,lambd)
    
    # maintain buffered merge scores to avoid recomputing, init first with all singleton merges allowed by graph
    buffered_merge_values = {}
    steps = 0
    value_steps = 0
    if interactions_graph is None:
        interactions_graph = nx.complete_graph(dataset.shape[1])
    for i in range(len(partition)):
        connections = interactions_graph.edges(i)

        if len(connections)>0:
            buffered_merge_values[(i,)] = {}
        for node_1,node_2 in connections:
            other_node = node_1 if node_2 == i else node_2
            if other_node < i:
                continue
            new_coalition = (node_1,node_2)
            coalition_sample = get_coalition_samples(new_coalition,instance, dataset, sampling_method, n_samples)
            predictions = model.predict(coalition_sample) if not is_classification else model.predict_proba(coalition_sample)[:,1]
            value_function = np.mean(predictions)- average_prediction
            buffered_merge_values[(i,)][(other_node,)] = value_function
            value_steps += 1
    
    # merge the coalitions: greedy merging until no more merges are possible either due to size or threshold
    merge_history = []
    merge_scores = []
    merge_history.append(partition.copy())
    merge_scores.append((component_sum-value_target)**2+regularization_term)
    best_score = merge_scores[0]
    
    while len(partition) > 1:
        best_merge = None
        for set_1 in buffered_merge_values.keys():
            for set_2 in buffered_merge_values[set_1].keys():
                merged_value = component_sum - partition_values[set_1] - partition_values[set_2] + buffered_merge_values[set_1][set_2]
                delta = 0
                #delta = abs(buffered_merge_values[set_1][set_2] - partition_values[set_1] - partition_values[set_2])
                
                new_regularization_term = regularization_term
                new_regularization_term = regularization_term - lasso_term([set_1,set_2],lambd) + lasso_term([set_1+set_2],lambd)
                if (value_target - merged_value)**2 + new_regularization_term - delta < best_score:
                    best_score = (value_target - merged_value)**2 + new_regularization_term - delta #* 1/2
                    best_merge = (set_1,set_2)
                steps += 1
        if best_merge is None:
            break
        else:
            # update scores and partition
            new_coalition = best_merge[0] + best_merge[1]
            partition.remove(best_merge[0])
            partition.remove(best_merge[1])
            partition.append(new_coalition)
            partition_values.pop(best_merge[0],None)
            partition_values.pop(best_merge[1],None)
            partition_values[new_coalition] = buffered_merge_values[best_merge[0]][best_merge[1]]

            regularization_term = lasso_term(partition,lambd)
            # update the sum of value functions
            component_sum = sum(partition_values.values())
            merge_history.append(partition.copy())
            best_score = (value_target - merged_value)**2 + regularization_term
            merge_scores.append(best_score)

            # update buffered merge values
            buffered_merge_values.pop(best_merge[0],None)
            buffered_merge_values.pop(best_merge[1],None)
            for set_1 in buffered_merge_values.keys():
                buffered_merge_values[set_1].pop(best_merge[0],None)
                buffered_merge_values[set_1].pop(best_merge[1],None)
            if len(new_coalition) < max_coalition_size:
                for set_1 in buffered_merge_values.keys():
                    if len(set_1) + len(new_coalition) > max_coalition_size:
                        continue
                    compute_merge = False
                    for element in set_1:
                        edges = interactions_graph.edges(element)
                        for node_1,node_2 in edges:
                            other_node = node_1 if node_2 == element else node_2
                            if other_node in new_coalition:
                                compute_merge = True
                                break
                        if compute_merge:
                            break
                    #compute_merge = is_subclique(interactions_graph,set_1+new_coalition)
                    if compute_merge:
                        coalition_sample = get_coalition_samples(new_coalition+set_1,instance, dataset, sampling_method, n_samples)
                        predictions = model.predict(coalition_sample) if not is_classification else model.predict_proba(coalition_sample)[:,1]
                        value_function = np.mean(predictions) - average_prediction
                        buffered_merge_values[set_1][new_coalition] = value_function
                        value_steps += 1

                buffered_merge_values[new_coalition] = {}
    if lambd > 0 and len(merge_history[-1])==1:
        merge_scores.pop()
    best = np.argmin(merge_scores)
    return merge_history[best], steps, value_steps

def score_partition(partition,model,dataset,n_samples,instance,avg_pred):
    fx = model.predict(instance.reshape(1,-1))-avg_pred
    approx = 0
    for coalition in partition:
        coalition = tuple(coalition)
        sample = get_coalition_samples(coalition,instance,dataset,"marginal",n_samples)
        value = np.mean(model.predict(sample))-avg_pred
        approx += value
    return (fx-approx)**2
    

def compute_explanation(model,average_prediction,instance,dataset,partition,is_classification,explanation_type,sampling_method,n_samples):
    if explanation_type == "value":
        explanantion_values = []
        for coalition in partition:
            coalition_sample = get_coalition_samples(coalition,instance, dataset, sampling_method, n_samples)
            predictions = model.predict(coalition_sample) if not is_classification else model.predict_proba(coalition_sample)[:,1]
            value_function = np.mean(predictions)-average_prediction
            explanantion_values.append(value_function)
        return list(zip(partition,explanantion_values))

    elif explanation_type == "shap":
        return compute_coalition_shapleys(partition, model,instance,dataset,is_classification,sampling_method,n_samples*10)
    else:
        raise ValueError("explanation_type must be either 'value' or 'shap'") 

# comparison function by using only singletons
def compute_singleton_values(model,instance,dataset,is_classification,sampling_method,n_samples):
    n_variables = len(instance)
    values = np.zeros(n_variables)
    mean_pred = np.mean(model.predict(dataset)) if not is_classification else np.mean(model.predict_proba(dataset)[:,1])
    for i in range(n_variables):
        modified_data = get_coalition_samples([i],instance,dataset,sampling_method,n_samples)
        if is_classification:
            values[i] = np.mean(model.predict_proba(modified_data)[:,1])-mean_pred
        else:
            values[i] = np.mean(model.predict(modified_data))-mean_pred
    return list(map(lambda x: ((x,),values[x]),range(n_variables)))

def binomial_weight_kernel(players,coalition):
    indices_to_consider = []
    for i in range(len(players)):
        member = players[i][0]
        indices_to_consider.append(member)
    player_matrix = np.array(coalition[:,indices_to_consider])
    
    numerator = np.zeros(player_matrix.shape[0])
    active_players = np.sum(player_matrix,axis=1)
    d = len(players)
    coef = special.binom(d,active_players)
    denominator = coef*active_players*(d-active_players)
    numerator[:] = d-1
    numerator[np.where(denominator == 0)] = 0
    denominator = np.maximum(denominator,1)
    weights = numerator/denominator
    return weights




def compute_coalition_shapleys(players, model,instance,dataset,is_classification,sampling_method,n_samples):
    if len(players) == 1:
        return [(players[0],model.predict(instance.reshape(1,-1))[0] if not is_classification else model.predict_proba(instance.reshape(1,-1))[0,1])]
    data, coalition = sample_data(players,instance, dataset, sampling_method, n_samples)
    #mean_pred = np.mean(model.predict(dataset)) if not is_classification else np.mean(model.predict_proba(dataset)[:,1])
    predictions = model.predict(data) if not is_classification else model.predict_proba(data)[:,1]
    #coalition_shapleys = np.zeros(len(players))
    #for i in range(len(players)):
    #    member = players[i][0]
    #    mask_in = coalition[:,member] == 1
    #    mask_out = coalition[:,member] == 0
    #    coalition_shapleys[i] = np.mean(predictions[mask_in]) - np.mean(predictions[mask_out])
    indices_to_consider = []
    for i in range(len(players)):
        member = players[i][0]
        indices_to_consider.append(member)
    player_matrix = np.array(coalition[:,indices_to_consider])
    model = LinearRegression()
    model.fit(player_matrix,predictions,sample_weight=binomial_weight_kernel(players,coalition))
    coalition_shapleys = model.coef_
    return list(map(lambda x: (players[x],coalition_shapleys[x]),range(len(players))))

def lasso_term(partition,lambd):
    c = np.array(list(map(lambda x: len(x),partition)))
    c = c*(c-1)/2
    return lambd*np.sum(c)

import time
def ishap(model,instance,dataset,is_classification=False,sampling_method="marginal",n_samples_interaction=50000,n_samples_partition=1000,max_coalition_size=-1,
          alpha_additivity=0.05,lambd=0.01, verbose=False,explanation_type="value",return_interaction_graph=False, return_steps=False, greedy=True, use_graph=True):
    average_prediction = np.mean(model.predict(dataset)) if not is_classification else np.mean(model.predict_proba(dataset)[:,1])
    t = time.time()
    if max_coalition_size == -1:
        max_coalition_size = len(instance)
    
    interaction_graph = None
    if use_graph:
        interaction_graph = find_interactions(model,average_prediction,instance,dataset,is_classification,sampling_method,n_samples_interaction,alpha_additivity)
    if verbose:
        print("Time to find interactions: {}".format(time.time()-t))
    t = time.time()

    if greedy:
        partition, steps, value_steps = find_partition_greedy(interaction_graph,average_prediction,model,instance,dataset,is_classification,sampling_method,n_samples_partition,max_coalition_size,lambd)
    else:
        partition, steps, value_steps = find_partition_exhaustive(interaction_graph,average_prediction,model,instance,dataset,is_classification,sampling_method,n_samples_partition,max_coalition_size,lambd)
        

    if verbose:
        print("Time to find partition: {}".format(time.time()-t))
        print(steps,"partitions tested", value_steps,"value functions evaluated")

    t = time.time()
    explanation = compute_explanation(model,average_prediction,instance,dataset,partition,is_classification,explanation_type,sampling_method,n_samples_partition)
    if verbose:
        print("Time to compute explanation: {}".format(time.time()-t))
    
    if return_steps:
        return explanation,steps, value_steps
    if return_interaction_graph:
        return explanation,interaction_graph
    else:
        return explanation