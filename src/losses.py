# RoboMMM-TCN v1.0 — Copyright (c) 2026 Roman Muravyev
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# For commercial use without AGPLv3 obligations, a commercial
# license is required. Contact: roman.muravev@gmail.com

# -*- coding: utf-8 -*-

"""
Module 1: Robust Loss Functions.
StableTweedieLoss — numerically stable Tweedie deviance in log-space.
AdaptiveMonotonicLoss — Lagrangian penalty for monotonicity in Sobolev spaces.
"""

import torch
import torch.nn as nn


class StableTweedieLoss(nn.Module):
    """
    Highly numerically stable implementation of Tweedie deviance loss function.
    Models compound Poisson-Gamma retail sales sell-out distributions with zero-inflation.
    Uses log-link function: mu = exp(predictions) to prevent float overflow and singular points.
    """
    def __init__(self, p=1.5, log_link=True, eps=1e-7):
        super().__init__()
        if not (1.0 < p < 2.0):
            raise ValueError("Tweedie index p must lie in the open interval (1.0, 2.0)")
        self.p = p
        self.log_link = log_link
        self.eps = eps

    def forward(self, predictions, targets):
        y = torch.clamp(targets, min=0.0)
        p = self.p

        if self.log_link:
            z = predictions
            term2 = y * torch.exp(z * (1.0 - p)) / (1.0 - p)
            term3 = torch.exp(z * (2.0 - p)) / (2.0 - p)
            term1 = torch.pow(y, 2.0 - p) / ((1.0 - p) * (2.0 - p))
            loss = 2.0 * (term1 - term2 + term3)
        else:
            mu = torch.clamp(predictions, min=self.eps)
            term1 = torch.pow(y, 2.0 - p) / ((1.0 - p) * (2.0 - p))
            term2 = y * torch.pow(mu, 1.0 - p) / (1.0 - p)
            term3 = torch.pow(mu, 2.0 - p) / (2.0 - p)
            loss = 2.0 * (term1 - term2 + term3)

        return torch.mean(loss)


class AdaptiveMonotonicLoss(nn.Module):
    """
    Custom Lagrangian loss function that penalizes negative gradients (violations of monotonicity).
    Supports multi-dimensional input tensors (2D, 3D, 4D) via Python ellipses '...'.
    Includes dynamic penalty weight annealing to guarantee convergence.
    """
    def __init__(self, base_loss_fn, lmbda=10.0, mono_indices=None, annealing_rate=1.05):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.lmbda = lmbda
        self.mono_indices = mono_indices if mono_indices is not None else []
        self.annealing_rate = annealing_rate
        self.step_counter = 0

    def step_annealing(self):
        """Linearly/exponentially increases penalty weight lambda during training"""
        self.lmbda *= self.annealing_rate
        self.step_counter += 1

    def forward(self, model, inputs, targets):
        inputs.requires_grad_(True)
        predictions = model(inputs)

        base_loss = self.base_loss_fn(predictions, targets)

        if not self.mono_indices:
            return base_loss, base_loss, torch.tensor(0.0, device=inputs.device)

        grads = torch.autograd.grad(
            outputs=predictions,
            inputs=inputs,
            grad_outputs=torch.ones_like(predictions),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradient_penalty = torch.tensor(0.0, device=inputs.device)

        for idx in self.mono_indices:
            channel_grads = grads[..., idx]
            negative_grads = torch.clamp(-channel_grads, min=0.0)
            gradient_penalty += torch.mean(torch.square(negative_grads))

        total_loss = base_loss + self.lmbda * gradient_penalty
        return total_loss, base_loss, gradient_penalty
