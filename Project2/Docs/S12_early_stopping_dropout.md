# Model Generalisation: Early Stopping and Dropout (KSB S12)

## Implementation

Two regularisation techniques are built into `aiml_08_train_segmentation_demo.py`:

**Dropout** — `nn.Dropout2d` at the end of every encoder/decoder block (zeroes whole feature channels, the standard approach for CNNs). Off by default.

```python
if dropout > 0:
    layers.append(nn.Dropout2d(p=dropout))
```

**Early stopping** — training halts if `val_loss` doesn't improve by `--min-delta` for `--patience` epochs; the best-`val_loss` checkpoint is restored before saving.

```python
improved = val_loss < (best_val_loss - args.min_delta)
if improved:
    best_val_loss, best_epoch, epochs_no_improve = val_loss, epoch, 0
    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
else:
    epochs_no_improve += 1
if args.patience > 0 and epochs_no_improve >= args.patience:
    break
model.load_state_dict(best_state_dict)   # restore best, not last, epoch
```

---

## Observed Impact (`lr1e-3_bs4_full_metrics_1`, baseline run — dropout off, no early stopping, 15/15 epochs)

| Epoch | Train loss | Val loss | Val IoU |
| :---: | :---: | :---: | :---: |
| 1 | 0.295 | 0.306 | 0.735 |
| 9 | 0.087 | 0.112 | 0.923 |
| 10 | 0.071 | **0.272** | 0.674 |
| **11 (best)** | 0.080 | **0.091** | 0.942 |
| 15 (final) | 0.073 | 0.093 | 0.954 |

- **Best epoch: 11** (val_loss 0.091, val_IoU 0.942) — with early stopping (`--patience 5`), training would have stopped at epoch 16 anyway (never triggered here) but the best-checkpoint logic would still have exported the epoch-11 weights instead of epoch-15's, avoiding the epoch-10 spike being the last word.
- **Val loss is volatile without regularisation**: it spikes to 0.272 at epoch 10 (IoU drops to 0.67) before recovering — a sign of overfitting/instability with dropout disabled. Train loss keeps falling smoothly throughout, decoupled from this val-loss noise.
- **Final train/val gap is small (0.020)** and IoU is highest (0.954) at epoch 15, so this run did not overfit badly overall — but the epoch-10 instability shows why monitoring `val_loss` (not just training loss) and checkpointing the best epoch matters, even when a run doesn't need to stop early.

**Next step:** re-run with `--dropout 0.2 --patience 5` on the same data/seed to test whether dropout removes the epoch-10 spike and whether early stopping still exits before epoch 15.

---

## Regularised Run (`lr1e-3_bs4_dropout02_patience5`, dropout 0.2, patience 5, seed 43)

| Epoch | Train loss | Val loss | Val IoU |
| :---: | :---: | :---: | :---: |
| 3 | 0.130 | **0.370** | 0.622 |
| 10 | 0.086 | 0.274 | 0.707 |
| **12 (best)** | 0.095 | **0.103** | 0.950 |
| 15 (final) | 0.079 | **0.400** | 0.723 |

- **Early stopping did not trigger** (`stopped_early: false`) — best epoch was 12, so only 3 non-improving epochs (13–15) had elapsed by the `--epochs 15` cap, short of the `patience=5` needed to stop.
- **The instability was not removed — it got worse.** Val loss still spikes sharply (0.370 at epoch 3, 0.274 at epoch 10, and a new worst spike of **0.400 at the final epoch 15**), and best val_loss (0.103) is higher than the baseline's best (0.091, epoch 11). Best val_IoU (0.950 at epoch 12) is close to but slightly below the baseline's peak (0.954).
- **Best-checkpoint logic still did its job**: despite the noisy final epochs, the exported model is epoch 12's weights (val_loss 0.103, IoU 0.950), not epoch 15's degraded ones (val_loss 0.400, IoU 0.723) — this is exactly the failure mode early stopping/checkpointing is meant to guard against.

**Interpretation:** at `dropout=0.2` this small model/dataset did not become more stable — likely because with only ~117k parameters and a small dataset, dropout adds variance without much overfitting to correct, and the different seed (43 vs. baseline) also contributes to run-to-run noise. `--patience 5` is too loose to catch this: it never triggers within a 15-epoch budget. Two follow-ups worth testing: (1) `--epochs 30 --patience 5` to let early stopping actually engage, and (2) a lower `--dropout 0.1` to check whether 0.2 is simply too aggressive for this model size.
