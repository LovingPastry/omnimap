# Fisher/NBV Notes

## Overview
This directory isolates Fisher-information-style next-best-view logic from `gs_backend.py`.
It contains:

- `legacy_fisher.py`: preserves the current score definition based on raw first-order gradients.
- `diag_fisher.py`: new implementation based on a diagonal Fisher approximation using `grad^2`.
- `hemisphere_field.py`: hemisphere sampling, interpolation, and color mapping.
- `visualization.py`: Open3D updates and artifact export.
- `debug.py`: tensor summaries and debug message construction.
- `interfaces.py`: shared protocol and result containers.

## What The Legacy Implementation Actually Computes
The old path does:

1. Render an image.
2. Backpropagate `ones_like(rendered_image)`.
3. Read parameter gradients for `xyz + opacity`.
4. Accumulate those raw gradients across history.
5. Score a view with:

`score_legacy = sum(cur_grad / (history_grad + lambda))`

This quantity is useful as a heuristic, but it is not a true Hessian and not a standard Fisher Information value.

## Why The Legacy Quantity Is Not A Hessian
Let `g = dI / dtheta`, where `I` is the rendered image and `theta` is the parameter vector.

The legacy code reads `g` directly after a backward pass. That is a first-order derivative.

A Hessian would require second derivatives:

`H_ij = d^2 L / (dtheta_i dtheta_j)`

The legacy implementation never differentiates gradients again, so it does not build a Hessian matrix.

## Why A Diagonal Fisher Approximation Uses `grad^2`
For least-squares or Gaussian-noise style observation models, Fisher Information is commonly written as:

`F = J^T W J`

where:

- `J` is the Jacobian of image observations with respect to parameters
- `W` is an observation weighting matrix

The diagonal approximation keeps only per-parameter energy:

`diag(F) ~= g^2`

with `g` representing the per-parameter first-order gradient.

This gives a more standard non-negative statistic:

- `cur_stat = g^2`
- `history_stat = sum_k g_k^2`
- `score_diag = sum(cur_stat / (history_stat + lambda))`

## Why `grad^2` Is Preferable Here
- It is non-negative.
- It avoids sign cancellation across views.
- It matches the diagonal Fisher / Gauss-Newton intuition.
- It usually gives more stable view-comparison behavior.

## Debug Fields
- `api_current`: the score returned by the evaluator for the current view.
- `exact_current`: the same score recomputed from stored tensors; should match `api_current`.
- `top10_contrib_ratio`: fraction of total score carried by the top 10 parameter contributions.
- `Direction consistency`: checks whether `theta/phi` and `camera_center` describe the same direction.
- `Top Fisher sample directions`: the strongest sampled hemisphere directions and their angles to the current view.

## Switching Algorithms
The main backend uses an explicit import switch.

Default legacy behavior:

```python
from gaussian.renderer.nbv.legacy_fisher import LegacyFisherEvaluator
```

Switch to diagonal Fisher:

```python
from gaussian.renderer.nbv.diag_fisher import DiagFisherEvaluator as LegacyFisherEvaluator
```

The backend code can stay unchanged because both evaluators expose the same interface.

## Naming Notes
- Legacy module keeps historical names such as `history_hessian` in its logs for continuity, even though it is based on raw gradients.
- Diagonal Fisher module uses `cur_stat/history_stat` naming to avoid implying a true Hessian.
