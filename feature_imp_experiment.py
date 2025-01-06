import numpy as np
import shap
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

import synthetic_experiments as se
import supplementary.ishap as ishap
import random

DATAPOINTS_PER_MODEL = 300
NUM_MODELS = 1

def run_experiment():
    X, var_set = se.syntheticData(100000, 20, 1, 5, 'uniform')
    model = se.MultModel(var_set)


    explainer = shap.Explainer(model.predict, X, nsamples=50000)

    ishap_values = []
    var_set_gradients = []
    shap_values = []
    var_gradients = []
    for _ in range(DATAPOINTS_PER_MODEL):
        x = X[np.random.randint(0, len(X))]
        ishap_answer = ishap.compute_coalition_shapleys(var_set, model, x, X, False, 'independent', 50000)
        gradients = model.var_set_grad(np.array([x]))
        for i, v in enumerate(var_set):
            gradients[i] = (gradients[i] * np.prod(np.array([x])[:,v],axis=1))
        shap_answer = explainer(np.array([x])).values
        var_grad = model.single_value_grad(np.array([x]))

        ishap_values.append(list(map(lambda x: x[1], ishap_answer)))
        var_set_gradients.append(gradients)
        shap_values.append(shap_answer.ravel())
        var_gradients.append(var_grad * x)
    ishap_values = np.array(ishap_values)
    var_set_gradients = np.array(var_set_gradients)
    shap_values = np.array(shap_values)
    var_gradients = np.array(var_gradients)

    v_set_correlation = {}
    
    for i, v in enumerate(var_set):
        v_set_correlation[tuple(v)] =  stats.pearsonr(ishap_values[:,i], var_set_gradients[:,i])[0]
    
    var_correlation = {}
    for i in range(var_gradients.shape[1]):
        var_correlation[i] = stats.pearsonr(shap_values[:,i], var_gradients[:,i])[0]
    
    return v_set_correlation, var_correlation



if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    best_correlation = 0
    best_v_set_correlation = {}
    best_var_correlation = {} 
    mean_correlations_v_set = []
    mean_correlations_vars = []
    for _ in range(NUM_MODELS):
        print(_)
        v_set_correlation, var_correlation = run_experiment()
        print(v_set_correlation)
        # plot in bar plot the correlations
        fig, ax = plt.subplots()
        for i,(label, value) in enumerate(v_set_correlation.items()):
            ax.bar(i, value, label=str(label))
        
        for i, (label, value) in enumerate(var_correlation.items(),start=len(v_set_correlation)+1):
            ax.bar(i, value, label=str(label))
        ax.legend()
        plt.show()

        pd.DataFrame({"var_set":list(v_set_correlation.keys()), "correlation":list(v_set_correlation.values())}).to_csv('var_set_correlation_example.csv')
        pd.DataFrame({"var":list(var_correlation.keys()), "correlation":list(var_correlation.values())}).to_csv('var_correlation_example.csv')

        mean_correlations_v_set.append(np.mean(list(v_set_correlation.values())))   
        mean_correlations_vars.append(np.mean(list(var_correlation.values())))

    # box plot of the mean correlations
    # plt.boxplot([mean_correlations_v_set, mean_correlations_vars]) 
    # plt.show()

    pd.DataFrame({"avg var set correlation":mean_correlations_v_set, "avg vars correlation":mean_correlations_vars}).to_csv('avrage_correlation_grad.csv')