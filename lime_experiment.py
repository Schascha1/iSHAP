import lime.lime_tabular
import numpy as np
from data_loaders import *
import itertools
from csv import writer

from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier, BaggingRegressor, BaggingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.datasets import load_diabetes, load_wine, load_breast_cancer, fetch_california_housing
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


def test_dataset(dataset,classification,name,modelname,tests,amount_of_points,savefig=True):
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

    preds = []
    mse_singletons = 0
    explainer = lime.lime_tabular.LimeTabularExplainer(X, discretize_continuous=False,mode="classification" if classification else "regression")
    for _ in range(tests):
        point_indices = np.random.choice(range(0,X.shape[0]),size=amount_of_points,replace=False)
        partitions = []
        explanations_singletons = []
        explanations_lime = []
        points = np.zeros((amount_of_points,X.shape[1]))
        intercept = 0
        for i, point in enumerate(point_indices):
            instance = X[point,:]
            
            e = explainer.explain_instance(instance,model.predict_proba if classification else model.predict,num_features=X.shape[1],num_samples=5000)
            #explanation.show_in_notebook(show_table=True)
            intercept = intercept + e.intercept[1]
            explanation = e.local_exp[1]
            r = []
            for var, coefficient in explanation:
                val = coefficient*instance[var]
                r.append(((var,),val))
            explanation = r
            explanations_lime.append(e)

            partition = [x[0] for x in explanation]
            partitions.append(partition)
            #print(e.intercept[1],explanation)
            explanations_singletons.append(explanation.copy())
            points[i,:] = instance
        intercept /= amount_of_points
        _, point, metadata = piece_together_point(partitions,points)
        true_fx = float(model.predict_proba(point.reshape(1,-1))[:,1] if classification else model.predict(point.reshape(1,-1)))

        best_pred = np.inf
        for e in explanations_lime:
            predicted_fx = e.intercept[1]
            for var, coefficient in e.local_exp[1]:
                predicted_fx += coefficient*point[var]
            if abs(true_fx - best_pred) > abs(predicted_fx - true_fx):
                best_pred = predicted_fx

        mse_singletons += (best_pred-true_fx)**2
        preds.append(true_fx)
    tss = np.sum((np.array(preds)-np.mean(preds))**2)
    r2_lime = 1-mse_singletons/tss
    mse_singletons /= tests
    with open("results/accuracy/lime_"+modelname+".csv","a") as f:
        writerobject = writer(f)
        writerobject.writerow([name,r2_lime])
        f.close()
    
    return
    

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
    test_dataset(dataset,classification,name,model,tests,amount_of_points,savefig=True)
    print(name + " done")
