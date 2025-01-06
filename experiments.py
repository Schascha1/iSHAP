import numpy as np
from data_loaders import *
from ishap import ishap, compute_singleton_values, compute_coalition_shapleys
import nshap
import itertools
from csv import writer

from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier, BaggingRegressor, BaggingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.datasets import load_diabetes, load_breast_cancer, fetch_california_housing
import matplotlib.pyplot as plt
import random

import sys



def piece_together_point(partitions,points):
    new_point = np.zeros(points.shape[1])
    vars_to_cover = set(range(points.shape[1]))
    origin = []
    sum_of_elements = sum(map(len,partitions))
    picker = list(range(sum_of_elements))
    index_to_partition = {}
    offset_partition = {}
    offset  =0
    trial = 0
    for i,partition in enumerate(partitions):
        offset_partition[i] = offset
        for c in range(offset,offset+len(partition)):
            index_to_partition[c] = i
        offset += len(partition)
    while len(vars_to_cover) > 0:
        random.shuffle(picker)
        viable = False
        for i in range(len(picker))[::-1]:
            pick = picker[i]
            partition_index = index_to_partition[pick]
            partition = partitions[partition_index]
            offset = offset_partition[partition_index]
            coalition = set(partition[pick-offset])
            if not coalition.issubset(vars_to_cover):
                del picker[i]
            else:
                new_point[list(coalition)] = points[partition_index,list(coalition)]
                origin.append((partition_index,coalition))
                vars_to_cover = vars_to_cover - coalition
                viable = True
                break
        if not viable:
            vars_to_cover = set(range(points.shape[1]))
            new_point = np.zeros(points.shape[1])
            origin = []
            picker = list(range(sum_of_elements))
            trial += 1
            if trial == 10:
                return False, None, None
    return True, new_point, origin

def interpolate_prediction(origin,explanations,average_pred,is_singleton,verbose):
    contribution = 0
    for i, o in enumerate(origin):
        explanation = explanations[o[0]]
        component = o[1]
        if is_singleton:
            for var in component:
                contribution += explanation[var][1]
                print("Adding contribution of "+str(explanation[var][1])+" from "+str(explanation[var][0])) if verbose else None
        else:
            for e in explanation:
                if set(e[0]) == component:
                    contribution += e[1]
                    print("Adding contribution of "+str(e[1])+" from "+str(e[0])) if verbose else None
                    break
    return contribution+average_pred

def evaluate_interpolation(name,model,X,classification,sampling_method="marginal",amount_of_points=2,max_coalition_size=3,tests=50,greedy=True,
    explanation_type="value",n_samples_interaction=50000,n_samples_partition=2000,alpha_additivity=0.01,verbose=False):
    average_pred = np.mean(model.predict_proba(X)[:,1]) if classification else np.mean(model.predict(X))
    mse_ishap = 0
    preds = []
    for _ in range(tests):
        successful = False
        while not successful:
            point_indices = np.random.choice(range(0,X.shape[0]),size=amount_of_points,replace=False)
            partitions = []
            explanations_coalitions = []
            points = np.zeros((amount_of_points,X.shape[1]))
            for i, point in enumerate(point_indices):
                instance = X[point,:]
                explanation = []
                while len(explanation)<2:
                    explanation = ishap(model,instance,X,is_classification=classification,verbose=False,alpha_additivity=0.01,explanation_type=explanation_type,greedy=greedy, use_graph=True,
                lambd=0.001,return_interaction_graph=False,n_samples_interaction=n_samples_interaction,n_samples_partition=n_samples_partition,max_coalition_size=4,sampling_method=sampling_method)
                partition = [x[0] for x in explanation]
                partitions.append(partition)
                explanations_coalitions.append(explanation)
                points[i,:] = instance
            successful, point, metadata = piece_together_point(partitions,points)
        interpol_1 = interpolate_prediction(metadata,explanations_coalitions,average_pred,False,verbose)
        true_fx = model.predict_proba(point.reshape(1,-1))[:,1] if classification else model.predict(point.reshape(1,-1))
        if verbose:
            print("true fx",true_fx,"interpol ishap",interpol_1)
        mse_ishap += (interpol_1-true_fx)**2
        preds.append(true_fx)
    
    tss = np.sum((np.array(preds)-np.mean(preds))**2)
    R2_ishap = 1-mse_ishap/tss

    mse_ishap /= tests
    
    return R2_ishap, mse_ishap

def evaluate_singletons(model,X,classification,sampling_method="marginal",amount_of_points=2,max_coalition_size=3,tests=50,
    explanation_type="value",n_samples_interaction=10000,n_samples_partition=2000,alpha_additivity=0.05,verbose=False):
    average_pred = np.mean(model.predict_proba(X)[:,1]) if classification else np.mean(model.predict(X))
    mse_singletons = 0
    preds = []
    for _ in range(tests):
        point_indices = np.random.choice(range(0,X.shape[0]),size=amount_of_points,replace=False)
        partitions = []
        explanations_singletons = []
        points = np.zeros((amount_of_points,X.shape[1]))
        for i, point in enumerate(point_indices):
            instance = X[point,:]
            explanation = None
            if explanation_type == "value":
                explanation = compute_singleton_values(model,instance,X,classification,sampling_method,n_samples_partition)
            elif explanation_type == "shap":
                explanation = compute_coalition_shapleys([(i,) for i in range(X.shape[1])],model,instance,X,classification,sampling_method,n_samples_partition)
            else:
                raise ValueError("Unknown explanation type "+str(explanation_type))
            partition = [x[0] for x in explanation]
            partitions.append(partition)
            explanations_singletons.append(explanation)
            points[i,:] = instance
            
        #print(metadata)
        _, point, metadata = piece_together_point(partitions,points)
        interpol = interpolate_prediction(metadata,explanations_singletons,average_pred,True,verbose)
        true_fx = model.predict_proba(point.reshape(1,-1))[:,1] if classification else model.predict(point.reshape(1,-1))
        if verbose:
            print("true fx",true_fx,"interpol singleton",interpol)
        mse_singletons += (interpol-true_fx)**2
        preds.append(true_fx)
    tss = np.sum((np.array(preds)-np.mean(preds))**2)
    R2_singletons = 1-mse_singletons/tss

    mse_singletons /= tests
    
    return R2_singletons, mse_singletons

def get_powerset(full_set,max_order):
    return set(itertools.chain.from_iterable(itertools.combinations(full_set, r) for r in range(1,min(len(full_set),max_order))))

def interpolate_nshap(metadata,nshaps,average_pred,n_variables,max_order):
    origin_set = {}
    for i in range(len(nshaps)):
        origin_set[i] = []
    for origin,coalition in metadata:
        origin_set[origin] = origin_set[origin] + list(coalition)
    for i,abc in origin_set.items():
        origin_set[i] = set(abc)
    full_set = set(range(n_variables))
    powerset = get_powerset(full_set,max_order)
    pred = average_pred
    for origin, coalition in origin_set.items():
        use_power_set = get_powerset(coalition,max_order)
        powerset = powerset - use_power_set
        current_nshap = nshaps[origin]
        for coalition in use_power_set:
            coalition = tuple(sorted(coalition))
            pred += current_nshap[coalition]
    pred2 = pred
    p = len(nshaps)
    for coalition in powerset:
        coalition = tuple(sorted(coalition))
        average_pred = 0
        for i in range(p):
            average_pred += nshaps[i][coalition]
        average_pred /= p
        pred2 += average_pred
    return pred, pred2


def evaluate_nshap(model,X,classification,tests=50,degree=4):
    vfunc = nshap.vfunc.interventional_shap(model.predict_proba if classification else model.predict, X, target=1 if classification else None, num_samples=2000)
    average_pred = np.mean(model.predict_proba(X)[:,1]) if classification else np.mean(model.predict(X))
    tss = 0
    mse = 0
    preds = []
    mse2 = 0
    for _ in range(tests):
        point_indices = np.random.choice(range(0,X.shape[0]),size=amount_of_points,replace=False)
        partitions = []
        explanations = []
        points = np.zeros((amount_of_points,X.shape[1]))
        for i, point in enumerate(point_indices):
            instance = X[point,:]
            n_shapley_values = nshap.n_shapley_values(instance, vfunc, n=degree)
            explanations.append(n_shapley_values)
            partition = [(i,) for i in range(X.shape[1])]
            partitions.append(partition)
            points[i,:] = instance
        _, point, metadata = piece_together_point(partitions,points)
        n_variables = X.shape[1]
        interpol, interpol2 = interpolate_nshap(metadata,explanations,average_pred,n_variables,degree)
        true_fx = model.predict_proba(point.reshape(1,-1))[:,1] if classification else model.predict(point.reshape(1,-1))

        mse += (interpol-true_fx)**2
        mse2 += (interpol2-true_fx)**2
        preds.append(true_fx)
    tss = np.sum((np.array(preds)-np.mean(preds))**2)
    R2 = 1-mse/tss
    R2_2 = 1-mse2/tss
    return R2, R2_2, mse/tests


def test_dataset(dataset,classification,name,modelname,tests,amount_of_points,savefig=False):
    dataset["data"] = StandardScaler().fit_transform(dataset["data"])
    X = dataset["data"]
    Y = dataset["target"]
    if modelname == "mlp":
        model = MLPClassifier() if classification else MLPRegressor()
    elif modelname == "rf":
        model = RandomForestClassifier() if classification else RandomForestRegressor()
    elif modelname == "gb":
        model = HistGradientBoostingClassifier() if classification else HistGradientBoostingRegressor()
    elif modelname == "knn":
        model = KNeighborsClassifier() if classification else KNeighborsRegressor()
    elif modelname == "svm":
        model = SVC(probability=True) if classification else SVR()
    elif modelname == "linear":
        model = LogisticRegression() if classification else LinearRegression()
    model.fit(X,Y)

    
    max_coalition_size = 3
    r2_nshap = 0
    r2_nshap_full = 0
    if name not in ["student","breast_cancer","life","adult"]:
        r2_nshap, r2_nshap_full, _ = evaluate_nshap(model,X,classification,tests=tests,degree=3)
    r2_ishap, mse_ishap = evaluate_interpolation(name,model,X,classification,verbose=False,explanation_type="value",tests=tests,amount_of_points=amount_of_points,max_coalition_size=max_coalition_size)
    
    r2_singleton_shap, mse_singleton_shap = evaluate_singletons(model,X,classification,explanation_type="shap",tests=tests,amount_of_points=amount_of_points)
    r2_ishap = np.squeeze(r2_ishap)
    r2_singleton_shap = np.squeeze(r2_singleton_shap)
    r2_nshap = np.squeeze(r2_nshap)
    r2_nshap_full = np.squeeze(r2_nshap_full)

    plt.figure()
    r2s = np.array([r2_ishap, r2_singleton_shap, r2_nshap, r2_nshap_full])
    print(name,r2s)
    plt.bar(0,r2_ishap,label="ishap")
    plt.bar(1,r2_singleton_shap,label="SHAP")
    plt.bar(2,r2_nshap,label="N-SHAP")
    plt.bar(3,r2_nshap_full,label="N-SHAP full")
    plt.xticks([0,1,2,3],["ishap","SHAP","N-SHAP","N-SHAP full"])
    plt.ylim(0,1)
    plt.ylabel("R2")
    plt.title("R2 " + name + " dataset")
    if savefig:
        plt.savefig("plots/"+name+".png")
    with open("results/accuracy/"+modelname+".csv","a") as f:
        writerobject = writer(f)
        writerobject.writerow([name,r2_ishap,r2_singleton_shap,r2_nshap, r2_nshap_full])
        f.close()

def load_data(name):
    if name == "breast_cancer":
        return load_breast_cancer(), True
    elif name == "california":
        return fetch_california_housing(download_if_missing=True), False
    elif name == "boston":
        return load_boston(), False
    elif name == "diabetes":
        return load_diabetes(), False
    elif name == "credit":
        return load_credit(), True
    elif name == "wages":
        return load_wages(), False
    elif name == "adult":
        return load_adult(), True
    elif name == "insurance":
        return load_insurance(), False
    elif name == "student":
        return load_student(), False
    elif name == "life":
        return load_life(), False
    elif name == "bike":
        return load_bike(), False
    else:
        raise ValueError("Unknown dataset "+str(name))
    
if __name__ == "__main__":
    name = sys.argv[1]
    model = sys.argv[2]
    dataset, classification = load_data(name)
    np.random.seed(42)
    tests = int(sys.argv[3])
    amount_of_points = int(sys.argv[4])
    print("Testing "+name+" dataset")
    test_dataset(dataset,classification,name,model,tests,amount_of_points,savefig=False)
    print(name + " done")
