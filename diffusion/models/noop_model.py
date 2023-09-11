# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""NoOpModel algorithm and class."""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torchmetrics import Metric

from composer.models.base import ComposerModel

class NoOpModel(ComposerModel):
    """No-op model used for performance measurements.

    Args:
        original_model (torch.nn.Module): Model to replace.
    """

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.Tensor([1.5]))

    def loss(self, outputs: torch.Tensor, batch):
        y = torch.randn_like(self.weight)
        return F.mse_loss(outputs, y)

    def forward(self, batch):
        input = torch.randn_like(self.weight)
        return self.weight * input

    def get_metrics(self, is_train: bool) -> Dict[str, Metric]:
        return {}

    def eval_forward(self, batch, outputs: Optional[Any] = None):
        return self.forward(batch)

    def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> None:
        pass