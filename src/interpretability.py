# RoboMMM-TCN v1.0 — Copyright (c) 2026 Roman Muravyev
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# For commercial use without AGPLv3 obligations, a commercial
# license is required. Contact: roman.muravev@gmail.com

# -*- coding: utf-8 -*-

"""
Module 3: Interpretability — Temporal-Adstock SHAP with Causal DAG restrictions.
CausalDAG — directed acyclic graph for causal lag structures of promo factors.
TemporalAdstockSHAPExplainer — causal-constrained Shapley value explainer.
"""

import math
import numpy as np
import torch


class CausalDAG:
    """
    Represents Directed Acyclic Graph describing causal lag structures of promotional factors.
    Used to calculate Markov Blankets and prevent physical out-of-distribution (OOD) coalitions.
    """
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.adjacency_matrix = np.zeros((num_nodes, num_nodes))

    def add_edge(self, u, v):
        """ u -> v represents causal impact (e.g. Adstock(t-1) -> Adstock(t)) """
        self.adjacency_matrix[u, v] = 1.0

    def get_markov_blanket(self, target_node):
        """ Returns set of parents, children, and spouses (other parents of children) """
        parents = np.where(self.adjacency_matrix[:, target_node] == 1.0)[0]
        children = np.where(self.adjacency_matrix[target_node, :] == 1.0)[0]
        spouses = []
        for child in children:
            other_parents = np.where(self.adjacency_matrix[:, child] == 1.0)[0]
            spouses.extend(other_parents)

        blanket = set(parents).union(set(children)).union(set(spouses))
        blanket.discard(target_node)
        return list(blanket)


class TemporalAdstockSHAPExplainer:
    """
    Causal-constrained SHAP Explainer.
    Locks dependencies between lag variables to block non-physical OOD configurations.
    """
    def __init__(self, model, background_data, causal_dag):
        self.model = model
        self.background_data = torch.tensor(background_data, dtype=torch.float32)
        self.causal_dag = causal_dag

    def compute_shap_values(self, x_instance):
        x = torch.tensor(x_instance, dtype=torch.float32)
        num_features = len(x_instance)
        shap_values = np.zeros(num_features)

        for i in range(num_features):
            blanket = self.causal_dag.get_markov_blanket(i)
            coalitions = [[]]
            for node in blanket:
                coalitions = [c + [node] for c in coalitions] + coalitions

            marginal_contribution = 0.0
            total_weight = 0.0

            for S in coalitions:
                S_with_i = S + [i]
                weight = (
                    math.factorial(len(S)) * 
                    math.factorial(num_features - len(S) - 1) / 
                    math.factorial(num_features)
                )

                val_S = self._evaluate_coalition(x, S)
                val_S_i = self._evaluate_coalition(x, S_with_i)

                marginal_contribution += weight * (val_S_i - val_S)
                total_weight += weight

            shap_values[i] = marginal_contribution / (total_weight if total_weight > 0 else 1.0)

        return shap_values

    def _evaluate_coalition(self, x_instance, coalition):
        """ Evaluates model prediction by substituting background features for nodes outside coalition """
        model_inputs = self.background_data.clone()
        for idx in coalition:
            model_inputs[..., idx] = x_instance[idx]

        if len(model_inputs.shape) == 2:
            model_inputs = model_inputs.unsqueeze(1)

        with torch.no_grad():
            preds = self.model(model_inputs)

        return torch.mean(preds).item()
