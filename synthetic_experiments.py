import autograd.numpy as np
from autograd import jacobian
import numpy.ma as ma
from ishap import ishap
import sys 
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, rand_score
import nshap
from csv import writer    
import time

class MultModel:
    def __init__(self,variable_sets) -> None:
        self.variable_sets = variable_sets
        self.params = []
        for vset in variable_sets:
            p = []
            for _ in vset:
                p.append(np.random.uniform(0.5,1.5))#*np.random.choice([-1,1]))
                #p.append(1)
            self.params.append(np.array(p))
        self.jacobians = []
        for i in range(max(map(len,variable_sets))):
            if i == 0:
                self.jacobians.append(jacobian(self.predict))
            else:
                pass
                # self.jacobians.append(jacobian(self.jacobians[-1]))

    def fit(self,X,Y):
        return
    
    def predict(self,X):
        Y = 0
        variable_sets = self.variable_sets
        for ind, v_set in enumerate(variable_sets):
            p = self.params[ind]
            inner_sum = np.prod(X[:,v_set]*p,axis=1)
            Y = Y + inner_sum   
        return Y
    
    def var_set_grad(self,X):
        if len(X) > 1:
            pass
            raise Exception("Only one instance allowed")
        results = []
        jr = []
        for j in self.jacobians:
            jr.append(j(X))
        for i, vset in enumerate(self.variable_sets):
            p = self.params[i]
            # tmp = jr[len(vset)-1][0] #TODO fix when more than one instance is allowed
            # for i in vset:
            #     tmp = tmp[0,i]
            results.append(np.prod(p))
        return results
    
    def single_value_grad(self,X):
        if len(X) > 1:
            pass
            raise Exception("Only one instance allowed")
        results = []
        jr = []
        for j in self.jacobians:
            jr.append(j(X))
        for vset in [[i] for i in range(max(map(max,self.variable_sets))+1)]:
            tmp = jr[len(vset)-1][0] #TODO fix when more than one instance is allowed
            for i in vset:
                tmp = tmp[0,i]
            results.append(tmp)
        return results



class SinSum:
    def __init__(self,variable_sets) -> None:
        self.variable_sets = variable_sets
        self.params = []
        for vset in variable_sets:
            p = []
            for _ in vset:
                p.append(np.random.uniform(0.5,1.5))
            self.params.append(np.array(p))
        self.jacobians = []
        for i in range(max(map(len,variable_sets))):
            if i == 0:
                self.jacobians.append(jacobian(self.predict))
            else:
                self.jacobians.append(jacobian(self.jacobians[-1]))

    def fit(self,X,Y):
        return
    
    def predict(self,X):
        Y = 0
        variable_sets = self.variable_sets
        for ind, v_set in enumerate(variable_sets):
            inner_sum = 0
            p = self.params[ind]
            inner_sum = np.sum(X[:,v_set]*p,axis=1)
            Y = Y + np.sin(inner_sum)
        return Y
    
    def var_set_grad(self,X):
        raise Exception("Delete this exceprion if you think this makes sense")
        if len(X) > 1:
            pass
            raise Exception("Only one instance allowed")
        results = []
        jr = []
        for j in self.jacobians:
            jr.append(j(X))
        for vset in self.variable_sets:
            tmp = jr[len(vset)-1][0] #TODO fix when more than one instance is allowed
            for i in vset:
                tmp = tmp[0,i]
            results.append(tmp)
        return results
    
    

def NMI(patterns_gt,patterns_pred,n_variables):
    labels_gt = np.zeros((n_variables+1,),dtype=int)
    labels_pred = np.zeros((n_variables+1,),dtype=int)
    l = 0
    for pattern in patterns_gt:
        for var in pattern:
            labels_gt[var] = l
        l += 1
    l = 0
    for pattern in patterns_pred:
        for var in pattern:
            labels_pred[var] = l
        l += 1
    if len(np.unique(labels_gt)) > 0 and len(np.unique(labels_pred)) > 0:
        labels_gt = labels_gt[:-1]
        labels_pred = labels_pred[:-1]
    score = normalized_mutual_info_score(labels_gt,labels_pred,average_method="arithmetic")
    return score

def score_partition(patterns_gt,patterns_pred,n_variables):
    labels_gt = np.zeros((n_variables+1,),dtype=int)
    labels_pred = np.zeros((n_variables+1,),dtype=int)
    l = 0
    for pattern in patterns_gt:
        for var in pattern:
            labels_gt[var] = l
        l += 1
    l = 0
    for pattern in patterns_pred:
        for var in pattern:
            labels_pred[var] = l
        l += 1
    if len(np.unique(labels_gt)) > 0 and len(np.unique(labels_pred)) > 0:
        labels_gt = labels_gt[:-1]
        labels_pred = labels_pred[:-1]
    score = adjusted_rand_score(labels_gt,labels_pred)
    return score


def syntheticData(n_samples,n_variables,p_used_variables,max_vars_per_set,data_generation):

    variables = np.array(list(range(n_variables)))
    utilize_variables = 0
    while np.sum(utilize_variables) == 0:
        utilize_variables = np.random.binomial(1,p_used_variables,(n_variables,))
        utilize_variables = ma.make_mask(utilize_variables)
    used_variables = list(variables[utilize_variables])
    variable_sets = []
    while len(used_variables) > 0:
        set_size = np.random.poisson(1.5)
        while set_size == 0 or set_size > max_vars_per_set:
            set_size = np.random.poisson(1.5)
        if set_size >= len(used_variables):
            variable_sets.append(used_variables)
            break
        else:
            S_i = list(np.random.choice(used_variables,size=set_size,replace=False))
            variable_sets.append(S_i)
            for v in S_i:
                used_variables.remove(v)
    X = np.zeros(shape=(n_samples,n_variables))
    if data_generation == "uniform":
        for i in range(n_variables):
            X[:,i] = np.random.uniform(0,3,size=(n_samples,))
    elif data_generation == "normal":
        for i in range(n_variables):
            X[:,i] = np.random.normal(loc=np.random.uniform(1,3),scale=np.random.uniform(0.5,1.5),size=(n_samples,))
    elif data_generation == "exponential":
        for i in range(n_variables):
            X[:,i] = np.random.exponential(scale=np.random.uniform(-1.5,1.5),size=(n_samples,))
    return X,variable_sets

def eval(patterns_gt,patters_pred,n_variables):
    precision = 0
    if len(patters_pred) == 0:
        return 1, 0 ,0
    for pred in patters_pred:
        if pred in patterns_gt:
            precision += len(pred)
    
    precision /= n_variables

    recall = 0
    for gt in patterns_gt:
        if gt in patters_pred:
            recall += len(gt)
    recall /= n_variables
    f1 = 0
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2*precision*recall/(precision + recall)
    return precision, recall, f1
    #return precision, recall, f1



def pair_recall(partition_gt,partition_pred,n_variables):
    patterns_gt = list(map(lambda x: tuple(x),partition_gt))
    patterns_pred = list(map(lambda x: tuple(x),partition_pred))
    n = 0
    score = 0
    for pattern in patterns_gt:
        if len(pattern) == 1:
            if pattern in patterns_pred:
                score += 1
            n += 1
        true_postives = 0
        false_negatives = 0
        for i in range(len(pattern)):
            for j in range(i+1,len(pattern)):
                
                false_negatives += 1
                for pattern2 in patterns_pred:
                    if pattern[i] in pattern2 and pattern[j] in pattern2:
                        true_postives += 1
                        false_negatives -= 1
                        break
        if true_postives+false_negatives == 0:
            continue
        n += 1
        score += true_postives/(true_postives+false_negatives)
    if n == 0:
        return 1
    score /= n
    return score

def pair_precision(patterns_gt,patterns_pred,n_variables):
    patterns_gt = list(map(lambda x: tuple(x),patterns_gt))
    patterns_pred = list(map(lambda x: tuple(x),patterns_pred))
    n = 0
    score = 0
    for pattern in patterns_pred:
        if len(pattern) == 1:
            if pattern in patterns_gt:
                score += 1
            n += 1
        true_postives = 0
        false_positives = 0
        for i in range(len(pattern)):
            for j in range(i+1,len(pattern)):
                
                false_positives += 1
                for pattern2 in patterns_gt:
                    if pattern[i] in pattern2 and pattern[j] in pattern2:
                        true_postives += 1
                        false_positives -= 1
                        break
        if true_postives+false_positives == 0:
            continue
        n += 1
        score += true_postives/(true_postives+false_positives)
    if n == 0:
        return 1
    score /= n
    return score

def nshap_pattern_search(model,X,instance,classification,max_order):
    vfunc = nshap.vfunc.interventional_shap(model.predict_proba if classification else model.predict, X, target=1 if classification else None, num_samples=2000)
    n_shapley_values = nshap.n_shapley_values(instance, vfunc, n=max_order)
    n_shapley_values = {k:abs(v) for k,v in n_shapley_values.items()}
    result  =[coalition for coalition,_ in sorted(n_shapley_values.items(), key=lambda x: x[1], reverse=True)]
    return result

def evaluate_nshap_patterns(coalition_order,gt_patterns,n_variables):
    variables = set()
    partition = []
    for coalition in coalition_order:
        if len(set(coalition).intersection(variables))>0:
            continue
        else:
            partition.append(set(coalition))
            variables = variables.union(set(coalition))
        if len(variables) == n_variables:
            break
    return eval(gt_patterns,partition, n_variables)+ (score_partition(gt_patterns,partition,n_variables),)

from visualization import show_interaction_graph

def test_synthetic_data(X_sampling,funcmodel,n_variables,n_repetitions,sample_size):
    
    f1s_greedy = []
    f1s_greedy_full = []
    f1s_exhaustive_constrained = 0
    f1s_exhaustive_full = 0
    pair_precision_f = 0
    pair_precision_fu = 0
    pair_precision_g = []
    pair_precisions_unconstrained = []
    
    
    f1_nshap = 0
    pair_precision_nshap = 0

    runtime_gu = []
    runtime_gc = []
    runtime_nshap = []
    runtime_eu = []
    runtime_ec = []
    steps_gu = []
    steps_gc = []
    steps_nshap = []
    steps_eu = []
    steps_ec = []
    value_gu = []
    value_gc = []
    value_nshap = []
    value_eu = []
    value_ec = []

    pair_recall_eu = []
    pair_recall_ec = []
    pair_recall_gu = []
    pair_recall_gc = []
    pair_recall_nshap = []

    for iter in range(n_repetitions):
        X,vset = syntheticData(sample_size, n_variables, 1, 3,X_sampling)
        gt_patterns = list(map(lambda x:set(x),vset))
        model = funcmodel(vset)
        instance = X[np.random.randint(0,sample_size),:]

        t = time.time()
        partition_greedy_unconstrained, s, v = ishap(model,instance,X,is_classification=False,sampling_method="independent",n_samples_interaction=50000,n_samples_partition=2000,max_coalition_size=4,
        alpha_additivity=0.01,lambd=0.005, verbose=False,explanation_type="shap",return_interaction_graph=False,greedy=True,use_graph=False, return_steps=True)
        runtime_gu.append(time.time()-t)
        value_gu.append(v)
        steps_gu.append(s)
        t = time.time()

        partition_greedy_constrained, s, v = ishap(model,instance,X,is_classification=False,sampling_method="independent",n_samples_interaction=50000,n_samples_partition=2000,max_coalition_size=4,
        alpha_additivity=0.01,lambd=0.005, verbose=False,explanation_type="shap",return_interaction_graph=False,greedy=True,use_graph=True, return_steps=True)
        runtime_gc.append(time.time()-t)
        steps_gc.append(s)
        value_gc.append(v)
        patterns_greedy_unconstrained = list(map(lambda x:set(x[0]),partition_greedy_unconstrained))
        patterns_greedy_constrained = list(map(lambda x:set(x[0]),partition_greedy_constrained))

        
        precision_greedy, recall_greedy, f1_greedy = eval(gt_patterns,patterns_greedy_constrained,n_variables)
        precision_unconstrained , recall_unconstrained, f1_unconstrained = eval(gt_patterns,patterns_greedy_unconstrained,n_variables)
        
        
        pair_precision_greedy = pair_precision(gt_patterns,patterns_greedy_constrained,n_variables)
        pair_precision_unconstrained = pair_precision(gt_patterns,patterns_greedy_unconstrained,n_variables)

        pair_recall_gu.append(pair_recall(gt_patterns,patterns_greedy_unconstrained,n_variables))
        pair_recall_gc.append(pair_recall(gt_patterns,patterns_greedy_constrained,n_variables))

        pair_precision_g.append(pair_precision_greedy)
        pair_precisions_unconstrained.append(pair_precision_unconstrained)
        
        f1s_greedy.append(f1_greedy)
        f1s_greedy_full.append(f1_unconstrained)


        if n_variables > 10:
            continue
        
        t = time.time()
        partition_exhaustive_unconstrained, s, v = ishap(model,instance,X,is_classification=False,sampling_method="independent",n_samples_interaction=20000,n_samples_partition=2000,max_coalition_size=4,
        alpha_additivity=0.01,lambd=0.005, verbose=False,explanation_type="shap",return_interaction_graph=False,greedy=False,use_graph=False, return_steps=True)
        runtime_eu.append(time.time()-t)
        value_eu.append(v)
        steps_eu.append(s)
        t = time.time()
        partition_exhaustive_constrained, s, v = ishap(model,instance,X,is_classification=False,sampling_method="independent",n_samples_interaction=20000,n_samples_partition=2000,max_coalition_size=4,
        alpha_additivity=0.01,lambd=0.005, verbose=False,explanation_type="shap",return_interaction_graph=False,greedy=False,use_graph=True, return_steps=True)
        steps_ec.append(s)
        value_ec.append(v)
        runtime_ec.append(time.time()-t)
        patterns_exhaustive = list(map(lambda x:set(x[0]),partition_exhaustive_constrained))
        patterns_exhaustive_unconstrained = list(map(lambda x:set(x[0]),partition_exhaustive_unconstrained))
        

        precision_full, recall_full, f1_full = eval(gt_patterns,patterns_exhaustive,n_variables)
        precision_full_unconstrained, recall_full_unconstrained, f1_full_unconstrained = eval(gt_patterns,patterns_exhaustive_unconstrained,n_variables)

        pair_precision_full = score_partition(gt_patterns,patterns_exhaustive,n_variables)
        pair_precision_full_unconstrained = score_partition(gt_patterns,patterns_exhaustive_unconstrained,n_variables)

        pair_recall_eu.append(pair_recall(gt_patterns,patterns_exhaustive_unconstrained,n_variables))
        pair_recall_ec.append(pair_recall(gt_patterns,patterns_exhaustive,n_variables))
        f1s_exhaustive_constrained += f1_full
        pair_precision_f += pair_precision_full
        f1s_exhaustive_full += f1_full_unconstrained
        pair_precision_fu += pair_precision_full_unconstrained

        t = time.time()
        results = nshap_pattern_search(model,X,instance,False,4)
        steps_nshap.append(len(results))
        value_nshap.append(len(results))
        runtime_nshap.append(time.time()-t)
        ps, rs, f1,nn = evaluate_nshap_patterns(results,gt_patterns,n_variables)
        pair_precision_nshap += nn
        f1_nshap = f1_nshap+ f1
        variables = set()
        patterns_nshap = []
        for coalition in results:
            if len(set(coalition).intersection(variables))>0:
                continue
            else:
                patterns_nshap.append(set(coalition))
                variables = variables.union(set(coalition))
            if len(variables) == n_variables:
                break
        pair_recall_nshap.append(pair_recall(gt_patterns,patterns_nshap,n_variables))

    f1_nshap = f1_nshap/n_repetitions
    pair_precision_nshap = pair_precision_nshap/n_repetitions
    f1s_exhaustive_constrained = f1s_exhaustive_constrained/n_repetitions
    pair_precision_f = pair_precision_f/n_repetitions
    f1s_exhaustive_full = f1s_exhaustive_full/n_repetitions
    pair_precision_fu = pair_precision_fu/n_repetitions
    return (f1s_exhaustive_constrained, f1s_exhaustive_full, np.mean(f1s_greedy), np.mean(f1s_greedy_full), f1_nshap, pair_precision_f, pair_precision_fu, np.mean(pair_precision_g), np.mean(pair_precision_unconstrained), pair_precision_nshap, 
    np.mean(runtime_ec), np.mean(runtime_eu), np.mean(runtime_gc), np.mean(runtime_gu), np.mean(runtime_nshap), np.mean(steps_ec), np.mean(steps_eu), np.mean(steps_gc), np.mean(steps_gu), np.mean(steps_nshap),
    np.mean(value_ec), np.mean(value_eu), np.mean(value_gc), np.mean(value_gu), np.mean(value_nshap),np.mean(pair_recall_ec), np.mean(pair_recall_eu), np.mean(pair_recall_gc), np.mean(pair_recall_gu),np.mean(pair_recall_nshap))

sample_variants = ["uniform","normal"]


if __name__ == "__main__":
    sampling = sys.argv[1]
    model_class = sys.argv[2]
    model = MultModel if model_class == "mult" else SinSum
    n_repetitions = int(sys.argv[3])
    np.random.seed(0)
    n_samples = 10000
    var_count = [4,6,8,10,15,20,30,50,75,100]
    savefig = False
    ff = []
    fg = []
    nf = []
    ng = []
    fu = []
    nu = []
    nshap_f1 = []
    nshap_pair_precision = []
    f1_total = []
    pair_precision_total = []
    for n_variables in var_count:
        (f1_full, f1_full_unconstrained, f1_greedy, f1_unconstrained,f1_nshap, pair_precision_full,pair_precision_full_unconstrained, pair_precision_greedy, pair_precision_unconstrained, pair_precision_nshap,
        runtime_ec, runtime_eu, runtime_gc, runtime_gu, runtime_nshap, steps_ec, steps_eu, steps_gc, steps_gu, steps_nshap,
        vsteps_ec, vsteps_eu, vsteps_gc, vsteps_gu, vsteps_nshap, pair_recall_full, pair_recall_full_unconstrained, pair_recall_greedy, pair_recall_greedy_unconstrained, pair_recall_nshap)= test_synthetic_data(sampling,model,n_variables,n_repetitions,n_samples)
        
        fg.append(f1_greedy)
        fu.append(f1_unconstrained)
        
        ng.append(pair_precision_greedy)
        nu.append(pair_precision_unconstrained)
        if n_variables <= 10:
            nshap_f1.append(f1_nshap)
            nshap_pair_precision.append(pair_precision_nshap)
            nf.append(pair_precision_full)
            ff.append(f1_full)
            f1_total.append(f1_full_unconstrained)
            pair_precision_total.append(pair_precision_full_unconstrained)
        with open("results/"+sampling+"-"+model_class+".csv","a") as f:
            writerobject = writer(f)
            writerobject.writerow([n_variables,f1_greedy,f1_unconstrained, pair_precision_greedy, pair_precision_unconstrained, runtime_gc, runtime_gu, steps_gc, steps_gu, vsteps_gc, vsteps_gu, pair_recall_greedy, pair_recall_greedy_unconstrained])
            f.close()
        with open("results/"+sampling+"-"+model_class+"-nshap.csv","a") as f:
            if n_variables <= 10:
                writerobject = writer(f)
                writerobject.writerow([n_variables,f1_full,f1_full_unconstrained,f1_nshap, pair_precision_full,pair_precision_full_unconstrained,pair_precision_nshap,
                                       runtime_ec, runtime_eu, runtime_nshap, steps_ec, steps_eu, steps_nshap, vsteps_ec, vsteps_eu, vsteps_nshap,pair_recall_full, pair_recall_full_unconstrained,pair_recall_nshap])
            f.close()
        print(sampling,model_class,n_variables,"done")
    
    
    plt.plot(var_count,fg,label="Greedy Graph")
    plt.plot(var_count,fu,label="Greedy Unconstrained")
    border = var_count.index(10)+1
    plt.plot(var_count[:border],ff,label="Exhaustive Graph")
    plt.plot(var_count[:border],f1_total,label="Exhaustive Unconstrained")
    plt.plot(var_count[:border],nshap_f1,label="NSHAP")
    plt.xlabel("Number of variables")
    plt.ylabel("F1 score")
    plt.ylim(0,1)
    plt.legend()
    if savefig:
        plt.savefig("plots/synthetic/f1_"+sampling+"_"+model_class+".png")

    plt.figure()
    
    plt.plot(var_count,ng,label="Greedy")
    plt.plot(var_count[:border],nshap_pair_precision,label="NSHAP")
    plt.plot(var_count[:border],nf,label="Full")
    plt.ylim(0,1)
    plt.legend()
    plt.xlabel("Number of variables")
    plt.ylabel("Adjusted Rand Index")
    if savefig:
        plt.savefig("plots/synthetic/rand_"+sampling+"_"+model_class+".png")
    
    