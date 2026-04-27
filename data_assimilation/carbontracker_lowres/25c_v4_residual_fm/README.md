# Phase 25c v4 — Residual Flow Matching (ArchesWeatherGen-style)

Two-stage training, faithful to AWG's recipe (arxiv 2412.12971).

## Phase 1: deterministic UNet backbone with rollout fine-tune

`phase1_det_backbone/train.py` — pure deterministic UNet, MSE loss.

```
python train.py --max-steps 30000                  # single-step pretraining (~3h on 1× A40)
python train.py --rollout-only --max-steps 8000    # rollout-FT phase (K=4 window, full grad)
```

Recipe details:
- Same hyperparameters as Phase 11 / Phase 24 best (Optuna study).
- `predict_delta=True` so the UNet predicts Δco2 rather than full state — easier MSE landscape.
- Single-step phase: standard MSE on `co2massmix_next`.
- Rollout phase: `n_timesteps=4`, all timesteps grad-enabled (`no_grad_step_shedule={"t_no_grad":[]}`),
  loss is MSE summed across rollout steps. Lit module's existing forward loop already chains
  `curr_data |= curr_preds` so the model's own outputs feed back as the next-step prior — exactly
  the AWG f_θ rollout-FT setup.
- (TODO) add quadratic discount `coeff_i = 1/(1+i)²` on per-step MSE — currently uniform across
  steps. To do this cleanly, implement `MSE` with `step_weights` kwarg in
  `neural_transport/tools/loss.py`.

## Phase 2: residual flow-matching head

`phase2_residual_fm/train.py` — TODO. Loads the Phase 1 best ckpt as a frozen `f_det`, then
trains an FM model `g_φ` to predict the residual `r = (x_next - f_det(x)) / σ` conditioned on
`[x_t, f_det(x_t)]`. Inference: `x_next = f_det(x_t) + σ · g_φ(noise | x_t, f_det(x_t))`.

Required code changes (not yet implemented):
- New wrapper `ResidualFlowMatching` in `neural_transport/models/flowmatching.py` that:
  - holds a frozen `f_det: RegularGridUNet` instance;
  - replaces `target_var = co2massmix_next` with `residual = (next - det) / sigma_res`;
  - feeds `[x_t, det_pred, residual_at_time_t, time]` into the velocity UNet;
  - postprocess: `pred_state = det_pred + sigma_res * generated_residual`.
- The `sigma_res` normalizer is one statistic computed ahead of training over the train split
  (`sigma_res = std(x_next - f_det(x))`).

## Why this is the right answer

For weather/climate, AWG show that *single-step FM cannot stably AR-rollout on its own*. Their
stability comes entirely from the deterministic anchor: `f_det` is already a strong (rollout-
fine-tuned) one-step predictor, so even on the AR trajectory the residuals stay in distribution.
The FM head only generates a tiny stochastic correction — its target distribution stays
narrow even as `f_det(x_t)` drifts.

Our Phase 25c v0/v1/v2/v3 are all attempts to make the FM model itself robust to its own AR
errors. AWG's empirical finding: that doesn't work — the cleaner separation is to train a
strong deterministic backbone first, then layer FM on top to model the (small) residual.

## Cost / status

- Phase 1 single-step: ~3h on one A40, then rollout phase ~2h. Output: `phase1_det_backbone/singlestep/checkpoints/best.ckpt`.
- Phase 2: ~3h training + 13min eval. Output: `phase2_residual_fm/singlestep/checkpoints/best.ckpt`.
- Total: ~8h end-to-end. Submit each phase separately (Phase 2 depends on Phase 1).

Status: Phase 1 train.py shipped (un-tested). Phase 2 design only.
