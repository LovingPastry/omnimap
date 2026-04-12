from __future__ import annotations

import torch

try:
    from .legacy_fisher import LegacyFisherEvaluator
except ImportError:
    from gaussian.renderer.nbv.legacy_fisher import LegacyFisherEvaluator


class DiagFisherEvaluator(LegacyFisherEvaluator):
    def _current_stat_from_raw_grad(self, raw_grad: torch.Tensor) -> torch.Tensor:
        return torch.square(raw_grad)

    def _debug_labels(self):
        return {
            "stat_label": "cur_stat",
            "history_label": "history_stat",
            "denominator_label": "history_stat + lambda",
            "contribution_label": "cur_stat/(history_stat+lambda)",
        }


class LogFisherEvaluator(LegacyFisherEvaluator):
    def _current_stat_from_raw_grad(self, raw_grad: torch.Tensor) -> torch.Tensor:
        return torch.log1p(raw_grad)

    def _debug_labels(self):
        return {
            "stat_label": "cur_stat",
            "history_label": "history_stat",
            "denominator_label": "history_stat + lambda",
            "contribution_label": "cur_stat/(history_stat+lambda)",
        }


class LogSquareFisherEvaluator(LegacyFisherEvaluator):
    def _current_stat_from_raw_grad(self, raw_grad: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.square(raw_grad))

    def _debug_labels(self):
        return {
            "stat_label": "cur_stat",
            "history_label": "history_stat",
            "denominator_label": "history_stat + lambda",
            "contribution_label": "cur_stat/(history_stat+lambda)",
        }
