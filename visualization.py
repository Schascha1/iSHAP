import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import networkx as nx
from utils import compute_value_function
import colorsys
import matplotlib.patches as mpatches

#plt.style.use('seaborn-whitegrid')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         "text.usetex": True}
plt.rcParams.update(params)

def plot_ishap(explanation_coalitions,explanation_singletons,instance,fx,feature_names,title="",ax=None,ylabel=""):
    if ax is None:
        _, ax = plt.subplots()
    coalition_sets = list(map(lambda x: x[0], explanation_coalitions))
    values_coalitions = list(map(lambda x: x[1], explanation_coalitions))
    order = np.argsort(np.abs(values_coalitions))[::-1]
    values_coalitions = np.array(values_coalitions)[order]
    coalition_sets = [coalition_sets[i] for i in order]

    values_singletons = np.array(list(map(lambda x: x[1], explanation_singletons)))
    instance = list(map(lambda x: x if type(x)==str else "{:.2f}".format(x), instance))
    coalitions = list(map(lambda x: list(map(lambda y: feature_names[y] + ":" + instance[y], x)), coalition_sets))
    for i in range(len(coalitions)):
        coalition = coalitions[i]
        result = ""
        added_chars = 0
        for n in coalition:
            result += n
            added_chars += len(n)
            if added_chars > 15 and n != coalition[-1]:
                result += "\n "
                added_chars = 0
            elif n != coalition[-1]:
                result += ","
        coalitions[i] = result
    #coalitions = list(map(lambda x: ",".join(x), coalitions))
    interaction_effects = []
    ymax = 0
    ymin = 0
    for i in range(len(coalition_sets)):
        coalition = list(coalition_sets[i])
        interaction_effects.append(values_coalitions[i] - np.sum(values_singletons[coalition]))

    total_sum = np.sum(np.abs(values_coalitions))
    intermediate_sum = 0
    ind = 0
    for i in range(len(values_coalitions)):
        intermediate_sum += np.abs(values_coalitions[i])
        if intermediate_sum >= 1*total_sum:
            ind = i
            break
    #ind = len(values_coalitions)
    values_coalitions = values_coalitions[:ind]
    coalitions = coalitions[:ind]

    has_positive = False
    has_negative = False
    for i in range(ind):
        vcoalition = values_coalitions[i]
        ieffect = interaction_effects[i]
        ax.bar(i, vcoalition, color='C0')
        if len(coalition_sets[i]) > 1:
            if ieffect > 0:
                ax.bar(i,ieffect,bottom=vcoalition-ieffect,color='green')
                has_positive = True
            else:
                ax.bar(i,ieffect,bottom=vcoalition-ieffect,color='red')
                has_negative = True
            ymax = max(ymax,vcoalition-ieffect)
            ymin = min(ymin,vcoalition-ieffect)
        ymax = max(ymax,vcoalition)
        ymin = min(ymin,vcoalition)
    ax.set_xticks(range(len(coalitions)),coalitions)
    ax.set_ylabel(ylabel)
    plt.gcf().autofmt_xdate()

    ax.set_title(title)
    ax.set_ylim([ymin*1.05,ymax*1.05])
    l = [mpatches.Patch(color='C0', label='Individual Effect')]
    if has_positive:
        l.append(mpatches.Patch(color='green', label='Positive Interaction'))
    if has_negative:
        l.append(mpatches.Patch(color='red', label='Negative Interaction'))
    ax.legend(handles=l)

    return

def plot_shapley(feature_names,instance,explanation_singletons,title="",ax=None):
    if ax is None:
        _, ax = plt.subplots()
    bot,top = ax.get_ylim()


    values_singletons = np.array(list(map(lambda x: x[1], explanation_singletons)))
    instance = list(map(lambda x: x if type(x)==str else "{:.2f}".format(x), instance))
    singletons = list(map(lambda x: feature_names[x] + ":" + instance[x], range(len(feature_names))))
    order = np.argsort(np.abs(values_singletons))[::-1]
    values_singletons = np.array(values_singletons)[order]
    colors = ['green' if x > 0 else 'red' for x in values_singletons]
    singletons = [singletons[i] for i in order]
    total_sum = np.sum(np.abs(values_singletons))
    intermediate_sum = 0
    ind = 0
    for i in range(len(values_singletons)):
        intermediate_sum += np.abs(values_singletons[i])
        if intermediate_sum > 1*total_sum:
            ind = i
            break
    #ind = len(values_singletons)
    values_singletons = values_singletons[:ind]
    singletons = singletons[:ind]
    # reorder list by order
    ax.bar(range(ind), values_singletons)
    ax.set_xticks(range(ind),singletons)
    plt.gcf().autofmt_xdate()
    ax.set_title(title)
    bot = min(bot,np.min(values_singletons))
    top = max(top,np.max(values_singletons))
    ax.set_ylim([bot*1.05,top*1.05])

def show_interaction_graph(graph,node_names,instance,pandas_instance,model, dataset,average_pred,is_classification,sampling,n_samples,ax=None):
    effects = {}
    for edge in graph.edges():
        i = edge[0]
        j = edge[1]
        vi = compute_value_function((i,),model,instance,average_pred,dataset,is_classification,sampling,n_samples)
        vj = compute_value_function((j,),model,instance,average_pred,dataset,is_classification,sampling,n_samples)
        vij = compute_value_function((i,j),model,instance,average_pred,dataset,is_classification,sampling,n_samples)
        graph[edge[0]][edge[1]]['weight'] = vij - vi - vj
        effects[(i,j)] = vij - vi - vj
    norm_factor = np.max(np.abs(list(effects.values())))
    for edge, value in effects.items():
        i = edge[0]
        j = edge[1]

        # 
        
        if value< 0:
            # make edge red and transparency proportional to value
            v = (abs(value)/norm_factor)**2
            rgb = colorsys.hsv_to_rgb(11/360, v, 0.8)
            rgb = tuple(map(lambda x: int(x*255),rgb))

            graph[edge[0]][edge[1]]['color'] = '#%02x%02x%02x' % rgb
        else:
            # make edge green and transparency proportional to value
            v = (abs(value)/norm_factor)**2
            rgb = colorsys.hsv_to_rgb(120/360, v, 0.8)
            rgb = tuple(map(lambda x: int(x*255),rgb))
            graph[edge[0]][edge[1]]['color'] = '#%02x%02x%02x' % rgb
    mapping = {}
    for i in range(len(node_names)):
        if type(pandas_instance[i]) == str:
            mapping[i] = node_names[i] + ":" + pandas_instance[i]
        else:
            mapping[i] = node_names[i] + ":" + "{:.2f}".format(pandas_instance[i])
    graph = nx.relabel_nodes(graph,mapping)
    
    A = nx.drawing.nx_agraph.to_agraph(graph)
    A.layout('dot')
    A.draw('graph.png')
    if ax is None:
        plt.figure()
        plt.axis("off")
        plt.imshow(plt.imread('graph.png'))
    else:
        ax.imshow(plt.imread('graph.png'))
        plt.axis("off")
    return