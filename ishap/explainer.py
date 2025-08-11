
import numpy as np
import networkx as nx
from ._utils import (
    sample_data, get_coalition_samples, lasso_term, generate_max_partition,
    compute_coalition_shapleys, test_additivity, compute_value_function
)
from .plots import interaction_graph_plot, partition_plot
import pandas as pd


class IShapExplanation:
    """A container for the results of an ISHAP explanation."""
    def __init__(self, shap_values, partition, partition_values, singleton_values,
                 instance, base_value, feature_names):
        self.shap_values = shap_values  # The final per-feature numpy array
        self.partition = partition
        self.partition_values = partition_values  # A dict mapping coalition -> value
        self.singleton_values = singleton_values  # A dict mapping singleton -> value
        self.instance = instance
        self.base_value = base_value
        self.feature_names = feature_names
    
    
    def to_table(self):
        """
        Returns a pandas DataFrame with columns: 'Coalition', 'Explanation Value', 'Interaction'.
        """

        data = []
        for coal in self.partition:
            features = [f"{self.feature_names[i]}:{self.instance[i]}" for i in coal]
            explanation_value = self.partition_values[coal]
            # Interaction: difference between coalition value and sum of singleton values
            singleton_sum = sum(self.singleton_values[(i,)] for i in coal)
            interaction = explanation_value - singleton_sum
            data.append({
                "Coalition": features,
                "Explanation Value": explanation_value,
                "Interaction": interaction
            })
            # sort by explanation value
        data.sort(key=lambda x: abs(x["Explanation Value"]), reverse=True)
        return pd.DataFrame(data)
    
    def __str__(self):
        """
        Returns a string representation of the explanation as a pandas DataFrame.
        """
        return self.to_table().__str__()
class IShapExplainer:
    """
    Computes Interaction-aware Shapley Values (ISHAP) for model explanations.
    """
    def __init__(self, model, background_data, is_classification=False, feature_names=None):
        self.model = model
        self.background_data = background_data
        self.is_classification = is_classification
        self.feature_names = feature_names if feature_names else [f'f_{i}' for i in range(background_data.shape[1])]
        
        if self.is_classification:
            self.average_prediction = self.model.predict_proba(self.background_data)[:, 1].mean()
        else:
            self.average_prediction = self.model.predict(self.background_data).mean()


    def explain(self, instance, **kwargs):
        """
        Generates a full ISHAP explanation for a single instance.
        
        This is the main method that computes partitions and values.
        
        Returns:
            IShapExplanation: An object containing all results.
        """
        # --- 1. Get parameters and find interactions ---
        use_graph = kwargs.get('use_graph', True)
        greedy = kwargs.get('greedy', True)
        n_samples_interaction = kwargs.get('n_samples_interaction', 50000)
        alpha_additivity = kwargs.get('alpha_additivity', 0.05)
        sampling_method = kwargs.get('sampling_method', 'marginal')
        
        interaction_graph = None
        if use_graph:
            interaction_graph = self._find_interactions(instance, sampling_method, n_samples_interaction, alpha_additivity)

        # --- 2. Find Partition ---
        n_samples_partition = kwargs.get('n_samples_partition', 1000)
        max_coalition_size = kwargs.get('max_coalition_size', len(instance))
        lambd = kwargs.get('lambd', 0.01)
        partition_finder = self._find_partition_greedy if greedy else self._find_partition_exhaustive
        
    
        
        final_partition = partition_finder(
            interaction_graph, instance, sampling_method, n_samples_partition,
            max_coalition_size, lambd
        )

        value_buffer = {}

        # --- 3. Compute all necessary values for plotting ---
        partition_values_dict = {}
        for coal in final_partition:
            if coal not in value_buffer:
                value_buffer[coal] = compute_value_function(
                    coal, self.model, instance, self.average_prediction, self.background_data,
                    self.is_classification, sampling_method, n_samples_partition
                )
            partition_values_dict[coal] = value_buffer[coal]
        # Ensure all singleton values are computed
        singleton_values_dict = {}
        for i in range(len(instance)):
            singleton_coal = (i,)
            if singleton_coal not in value_buffer:
                 value_buffer[singleton_coal] = compute_value_function(
                    singleton_coal, self.model, instance, self.average_prediction, self.background_data,
                    self.is_classification, sampling_method, n_samples_partition
                )
            singleton_values_dict[singleton_coal] = value_buffer[singleton_coal]
            
        # --- 4. Format output and return Explanation object ---
        explanation_values = self._compute_explanation(
            instance, final_partition, kwargs.get('explanation_type', 'shap'),
            sampling_method, n_samples_partition
        )
        shap_values_array = self._format_explanation(explanation_values)
        

        return IShapExplanation(
            shap_values=shap_values_array,
            partition=final_partition,
            partition_values=partition_values_dict,
            singleton_values=singleton_values_dict,
            instance=instance,
            base_value=self.average_prediction,
            feature_names=self.feature_names
        )
    
    

    def shap_values(self, X, **kwargs):
        """
        Computes the ISHAP values for a single instance or multiple instances.

        Args:
            X (np.ndarray): A single instance (1D) or multiple instances (2D) to explain.
            **kwargs: Arguments passed to the `explain` method.

        Returns:
            np.ndarray: An array of SHAP values. Shape is (n_features,) for a single
                        instance or (n_instances, n_features) for multiple instances.
        """
        if X.ndim == 1:
            explanation_tuples = self.explain(X, **kwargs)
            return self._format_explanation(explanation_tuples)
        else:
            results = [self.explain(X[i], **kwargs) for i in range(X.shape[0])]
            return np.array([self._format_explanation(res) for res in results])


    
    def _predict(self, data):
        """Internal predict function to handle classification and regression."""
        if self.is_classification:
            return self.model.predict_proba(data)[:, 1]
        return self.model.predict(data)

    def _find_interactions(self, instance, sampling_method, n_samples, alpha):
        """Builds the interaction graph based on additivity tests."""
        players = [(i,) for i in range(len(instance))]
        data, coalition = sample_data(players, instance, self.background_data, sampling_method, n_samples)
        predictions = self._predict(data)

        interaction_graph = nx.Graph()
        interaction_graph.add_nodes_from(range(len(players)))
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                if test_additivity(predictions, coalition, i, j, alpha):
                    interaction_graph.add_edge(i, j)
        return interaction_graph

    def _find_partition_greedy(self, G, instance, sampling_method, n_samples, max_size, lambd):
        """Finds the best partition using a greedy merge algorithm."""
        partition = [(i,) for i in range(len(instance))]
        value_target = self._predict(instance.reshape(1, -1))[0] - self.average_prediction
        
        partition_values = {}
        component_sum = 0
        for coal in partition:
            val = np.mean(self._predict(get_coalition_samples(coal, instance, self.background_data, sampling_method, n_samples))) - self.average_prediction
            partition_values[coal] = val
            component_sum += val

        lambd_reg = lambd * (component_sum - value_target)**2
        best_score = float('inf')
        while len(partition) > 1:
            best_merge = None
            best_value = None

            for i in range(len(partition)):
                for j in range(i + 1, len(partition)):
                    if G is not None and not G.has_edge(i, j):
                        continue
                    
                    merged_coal = tuple(sorted(partition[i] + partition[j]))
                    if len(merged_coal) > max_size:
                        continue

                    value_merge = compute_value_function(
                        merged_coal, self.model, instance, self.average_prediction,
                        self.background_data, self.is_classification, sampling_method, n_samples
                    )
                    
                    new_value = component_sum - partition_values[partition[i]] - partition_values[partition[j]] + value_merge
                    objective = (new_value - value_target)**2 + lasso_term([merged_coal], lambd_reg)
                    
                    if objective < best_score:
                        best_score = objective
                        best_merge = (i, j, merged_coal)
                        best_value = value_merge
            
            if best_merge is None:
                break
            
            i, j, merged_coal = best_merge
            partition_values[merged_coal] = best_value 
            component_sum -= (partition_values[partition[i]] + partition_values[partition[j]])
            component_sum += best_value   
            del partition_values[partition[i]]
            del partition_values[partition[j]]
            partition[i] = merged_coal        
            del partition[j]

        return partition
    
    def _find_partition_exhaustive(self, G, instance, sampling_method, n_samples, max_size, lambd):
        """Finds the best partition using an exhaustive search."""
        d = len(instance)
        max_partition = [list(range(d))]
        if G is not None:
            max_partition = [list(c) for c in nx.connected_components(G)]

        value_target = self._predict(instance.reshape(1,-1))[0] - self.average_prediction
        value_function_buffer = {}
        
        # Initial singleton values
        component_sum = 0
        for coal in [(i,) for i in range(d)]:
            val = np.mean(self._predict(get_coalition_samples(coal, instance, self.background_data, sampling_method, n_samples))) - self.average_prediction
            value_function_buffer[coal] = val
            component_sum += val

        lambd_reg = lambd * (component_sum - value_target)**2
        
        all_partitions = generate_max_partition(max_partition)
        top_partition, top_score = None, float('inf')

        for part in all_partitions:
            part = [tuple(p) for p in part]
            if any(len(c) > max_size for c in part):
                continue
            
            partition_value = 0
            for coal in part:
                if coal not in value_function_buffer:
                    value_function_buffer[coal] = np.mean(self._predict(get_coalition_samples(coal, instance, self.background_data, sampling_method, n_samples))) - self.average_prediction
                partition_value += value_function_buffer[coal]
            
            objective = (partition_value - value_target)**2 + lasso_term(part, lambd_reg)
            if objective < top_score:
                top_score, top_partition = objective, part
        
        return top_partition

    def _compute_explanation(self, instance, partition, explanation_type, sampling_method, n_samples):
        """Computes the final explanation values for the given partition."""
        if explanation_type == "value":
            explanation_values = []
            for coal in partition:
                val = np.mean(self._predict(get_coalition_samples(coal, instance, self.background_data, sampling_method, n_samples))) - self.average_prediction
                explanation_values.append(val)
            return list(zip(partition, explanation_values))
        
        elif explanation_type == "shap":
            return compute_coalition_shapleys(partition, self.model, instance, self.background_data, self.is_classification, sampling_method, n_samples)
        else:
            raise ValueError("explanation_type must be 'value' or 'shap'")

    def _format_explanation(self, explanation_tuples):
        """Converts the (coalition, value) list to a SHAP-like numpy array."""
        num_features = self.background_data.shape[1]
        output = np.zeros(num_features)
        
        if not explanation_tuples:
            return output
            
        for coalition, value in explanation_tuples:
            attribution = value / len(coalition)
            for feature_idx in coalition:
                output[feature_idx] += attribution
        return output




    def interaction_plot(self, instance, **kwargs):
        """
        Finds and plots the interaction graph for a single instance.

        Args:
            instance (np.ndarray): The single instance (1D array).
            **kwargs: Arguments passed to `_find_interactions`.
        """
        # Set default sampling if not provided
        n_samples = kwargs.get('n_samples_partition', 1000)
        alpha = kwargs.get('alpha_additivity', 0.05)
        sampling = kwargs.get('sampling_method', 'marginal')

        graph = self._find_interactions(instance, sampling, n_samples, alpha)
        # get interaction values for all edges
        interaction_values = {}

        for u, v in graph.edges():
            value_u = compute_value_function(
                (u,), self.model, instance, self.average_prediction, self.background_data,
                self.is_classification, sampling, n_samples
            )
            value_v = compute_value_function(
                (v,), self.model, instance, self.average_prediction, self.background_data,
                self.is_classification, sampling, n_samples
            )
            interaction_value = compute_value_function(
                (u, v), self.model, instance, self.average_prediction, self.background_data,
                self.is_classification, sampling, n_samples
            )
            interaction_values[(u, v)] = interaction_value - (value_u + value_v)
        interaction_graph_plot(graph, instance, interaction_values, self.feature_names)

    def partition_plot(self, instance, **kwargs):
        """
        Plots the final partition, showing each coalition's value and interaction score.
        
        Args:
            instance (np.ndarray): The single instance (1D array) to explain and plot.
            **kwargs: Arguments passed to the `explain` method.
        """
        explanation = self.explain(instance, **kwargs)
        partition_plot(explanation)