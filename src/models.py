# RoboMMM-TCN v1.0 — Copyright (c) 2026 Roman Muravyev
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# For commercial use without AGPLv3 obligations, a commercial
# license is required. Contact: roman.muravev@gmail.com

# -*- coding: utf-8 -*-

"""
Module 2: Monotonic Deep Learning.
CausalDilatedConv1D — causal dilated 1D convolution (no future leakage).
TemporalBlock — deep residual block with GELU, weight normalization, dropout.
TemporalConvolutionalNetwork — full TCN architecture for lag-aware MMM forecasting.
"""

import torch
import torch.nn as nn


class CausalDilatedConv1D(nn.Module):
    """
    Causal Dilated 1D Convolutional Layer.
    Ensures that the output at time t depends only on inputs before time t (no leaks from future).
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """
    Deep Causal Residual block for Temporal Convolutional Networks.
    Includes causal convolutions, GELU activation, Weight Normalization, and Spatial Dropout.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            CausalDilatedConv1D(n_inputs, n_outputs, kernel_size, dilation=dilation).conv
        )
        self.padding1 = (kernel_size - 1) * dilation
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            CausalDilatedConv1D(n_outputs, n_outputs, kernel_size, dilation=dilation).conv
        )
        self.padding2 = (kernel_size - 1) * dilation
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.act1, self.drop1, self.conv2, self.act2, self.drop2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.act_out = nn.GELU()

    def forward(self, x):
        out1 = self.conv1(x)
        if self.padding1 > 0:
            out1 = out1[:, :, :-self.padding1]
        out1 = self.act1(out1)
        out1 = self.drop1(out1)

        out2 = self.conv2(out1)
        if self.padding2 > 0:
            out2 = out2[:, :, :-self.padding2]
        out2 = self.act2(out2)
        out2 = self.drop2(out2)

        res = x if self.downsample is None else self.downsample(x)
        return self.act_out(out2 + res)


class TemporalConvolutionalNetwork(nn.Module):
    """
    Full Monotonic Temporal Convolutional Network (TCN) architecture.
    Provides robust, lag-aware time-series forecasting for multi-channel MMM.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1,
                dilation=dilation_size, dropout=dropout
            )]

        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        output = self.tcn(x_t)
        output_last = output.transpose(1, 2)[:, -1, :]
        return self.linear(output_last)
