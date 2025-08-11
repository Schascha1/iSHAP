# ishap/plots.py

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import colorsys
import matplotlib.patches as mpatches

# Set some default plot parameters for a clean look
plt.style.use('seaborn-v0_8-whitegrid')
params = {
    'legend.fontsize': 'large',
    'figure.figsize': (12, 5),
    'axes.labelsize': 'large',
    'axes.titlesize': 'x-large',
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'large',
}
plt.rcParams.update(params)


def interaction_graph_plot(graph, instance, interaction_values, feature_names=None, ax=None):
    """
    Visualizes the interaction graph.

    Args:
        graph (networkx.Graph): The interaction graph from the explainer.
        feature_names (list): List of feature names for node labels.
        ax (matplotlib.axes.Axes): Matplotlib axis to plot on.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Color edge blue/orange based on interaction value and adjust thickness based on magnitude
    # Use a coolwarm colormap for edge colors based on interaction value
    max_interaction = max(abs(val) for val in interaction_values.values())
    norm = plt.Normalize(vmin=-max_interaction, vmax=max_interaction)
    cmap = plt.cm.viridis
    edge_colors = [cmap(norm(interaction_values[edge])) for edge in graph.edges()]
    max_interaction = max(abs(val) for val in interaction_values.values())
    edge_widths = [2 * abs(interaction_values[edge]) / max_interaction for edge in graph.edges()]
    if feature_names:
        labels = {i: f"{name}:{instance[i]:.02f}" for i, name in enumerate(feature_names)}
        graph = nx.relabel_nodes(graph, labels)

    pos = nx.planar_layout(graph)  # Use planar layout for better readability
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=2000,
            width=2, font_size=10, ax=ax)
    nx.draw_networkx_edges(graph, pos, width=edge_widths, ax=ax, edge_color=edge_colors)

    # give colorbar for edge colors
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=-max_interaction, vmax=max_interaction))
    # Make the colorbar smaller
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.04)
    sm.set_array([])
    cbar.set_label('Interaction Effect on $f(x)$', rotation=270, labelpad=20)
    plt.tight_layout()

    ax.set_title("Feature Interaction Graph")
    plt.show()

def partition_plot(explanation, max_display=15, ax=None):
    """
    Creates a bar plot showing the value of each coalition in the final partition.

    Each bar is broken down into the sum of individual feature effects and the
    interaction effect.

    Args:
        explanation (IShapExplanation): The explanation object from explainer.explain().
        max_display (int): Maximum number of coalitions to display.
        ax (matplotlib.axes.Axes): Matplotlib axis to plot on.
    """
    partition = explanation.partition
    partition_values = explanation.partition_values
    singleton_values = explanation.singleton_values
    feature_names = explanation.feature_names
    instance = explanation.instance

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))

    # --- Data Preparation ---
    plot_data = []
    for coal in partition:
        coal_val = partition_values[coal]
        
        # Sum of individual effects for features in the coalition
        sum_of_singles = sum(singleton_values.get((f,), 0) for f in coal)
        
        # Interaction is the difference
        interaction_val = coal_val - sum_of_singles
        
        plot_data.append({
            'coalition': coal,
            'total_value': coal_val,
            'base_effects': sum_of_singles,
            'interaction': interaction_val
        })

    # Sort by absolute total value for visual impact
    plot_data.sort(key=lambda x: abs(x['total_value']), reverse=True)
    
    # Limit number of coalitions displayed
    if len(plot_data) > max_display:
        plot_data = plot_data[:max_display]
    
    plot_data.reverse() # Reverse for horizontal bar plot (largest on top)

    # --- Plotting ---
    coalition_labels = []
    for data in plot_data:
        # Create readable labels for coalitions
        label = "{" + "\n".join([f"{feature_names[i]}:{instance[i]:.02f}" for i in data['coalition']]) + "}"
        coalition_labels.append(label)

    base_effects = np.array([d['base_effects'] for d in plot_data])
    interactions = np.array([d['interaction'] for d in plot_data])
    
    y_pos = np.arange(len(plot_data))

    # Plot the base effects (sum of singletons)
    ax.barh(y_pos, base_effects, align='center', label='Individual Effect', color='gray')
    
    # Plot the interaction effects on top
    ax.barh(y_pos, interactions, left=base_effects, align='center', label='Interaction Effect', color='#337BFF')

    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coalition_labels, fontsize='medium')
    ax.set_xlabel("Contribution to model output")
    ax.set_title("iSHAP Partition Plot")
    ax.legend()
    # get the current axis limits
    xmin,xmax = ax.get_xlim()
    limit = max(abs(xmin), abs(xmax))
    # set the x-axis limits to be symmetric around zero and scale by 1.05
    ax.set_xlim(-1.05 * limit, 1.05 * limit)
    plt.tight_layout()
    plt.show()