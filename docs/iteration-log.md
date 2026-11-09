# Iteration log: making `generate.py` produce recognizable icons

Goal: simple prompts ("coffee cup", "heart", "house") come out as recognizable icons in the chosen set's style.
Each entry: observation -> diagnosis -> change -> result. Newest at the bottom.

## Iter 0: v1 baseline (2026-11-09)
- Run: 20-epoch plan, batch 16, 42,840 steps on RTX 4080. Best ckpt step 17,000, val 0.5735. Heavy overfit afterwards (train 0.03, val 0.93).
- Observation: samples reproduce the set's style (Lucide) but names do not yield recognizable content.
- Diagnosis (`scratchpad/diag.py`: loss with text / style conditioning dropped, 40 batches):

| split | full | no text | no style | neither |
|---|---|---|---|---|
| train | 0.2255 | 0.2341 | 0.2611 | 0.2766 |
| inset_test | 0.5681 | 0.5789 | 0.6058 | 0.6243 |

  The name is worth ~2% of loss, so the model ignores it. Style is worth ~6%.
- Likely cause: MiniLM vectors are unit-norm (~0.05 per dim) vs token embeddings (~1.0 per dim), and the name is one prefix token among 10.
- Change: `model.py` scales text_emb by sqrt(384) and projects it to `N_TEXT_TOKENS = 4` prefix slots (null_text is now [4, d]; prefix_len = 1 + 4 + K). 68 tests pass.
- Next: 3,000-step probe (`runs/v2probe`), rerun diag, compare the text gain.

## Iter 1: probe of the text rescale (3,000 steps, `runs/v2probe`)
- Result: text gain on inset_test still ~1% (0.8524 full vs 0.8595 no-text), and also ~2% on the first 40 tokens only. Rescaling alone did not fix it. Data checked: embeddings are unit-norm and aligned with names; 86% of held-out names also occur in other sets' training data (e.g. `heart` x15, `coffee` x8), so concepts are learnable.
- Old final ckpt (v1 step 42,840) on train: no-text 0.043 vs full 0.021, so the model uses the name to *memorize* training icons but does not generalize name -> shape. Overfitting/memorization is the core problem.
- v1 (cfg 1) samples on Lucide: right stroke style, but fragmentary shapes, not coherent icons.
- Changes: `ModelConfig.n_text_tokens` / `text_scale` (defaults keep v1 checkpoints loadable); classifier-free guidance on the name in `generate.py --cfg`; horizontal-flip augmentation (`flip_x`, `--p-flip`) on target+exemplars; `--dropout`.
- Next: v2 run, 8 epochs (v1's best was at ~epoch 8 and it overfit after), flip 0.5, dropout 0.1, 4 text tokens, text_scale 19.6.

## Iter 2: v2 run (`runs/v2`, 8 epochs, flip 0.5, dropout 0.1, 4 text tokens, text_scale 19.6)
- Training: 17,136 steps, ~40 min. Val 0.4934 (v1 best 0.5735); train 0.35, so no overfit gap and val still falling at the end.
- Name gain on inset_test: full 0.5038 vs no-text 0.5440 (~8%; v1 was ~2%). Train: 0.2706 vs 0.3294.
- Samples (Lucide, seed 0): `heart` recognizable at cfg 1 and cleaner at cfg 3; `house` appears occasionally at cfg 1; `coffee` not recognizable. cfg 3 turns non-heart prompts into scribbles (over-guidance).
- Next: sweep cfg 1.5-2 at lower temperature on more common concepts; if coffee still fails, train longer / bigger text signal.
- Sweep (temp 0.6): cfg 2 gives clean `heart` x4/6 and `house` x6/6 in Lucide style; `coffee`, `coffee cup`, `star` (19 training copies!), `key` still fail. cfg 1.5 is weaker. Since even data-rich `star` fails and v2 val was still falling, the model is undertrained.
- Changes: trainer now saves `ckpt_best.pt` / `ckpt_best.txt` at the lowest val.
- Next: v3 = 20 epochs with the same augmentation, `--p-drop-text 0.2` (more unconditional training, so CFG is sharper).

## Iter 3: v3 run (`runs/v3`, 20 epochs, p_drop_text 0.2), mid-run peek
- Step 13,000: best val 0.5384 (v2 had 0.538 near the same step, with lr already decaying; v3 is still at high lr).
- Peek samples (cfg 2, temp 0.6): `coffee` now gives mug/cup outlines in 2/6; `star` gives real stars but drawn tiny; `heart` good; `key` still wrong.
- Final v3: best val 0.4942 at step 21,000 (~epoch 10), then val rose (0.5126 at 26k) while train fell to 0.23, so the run was stopped at 27k. Overfitting again starts near epoch 10.
- v3 best samples (cfg 2): `heart` 5/6 (some open), `house` worse than v2, `coffee`/`star`/`key` wrong. No clear win over v2: longer training alone just memorizes.

## Iter 4: scale/shift augmentation (`runs/v4`)
- Diagnosis: 34k training icons are memorized by ~epoch 10; flips only double the data. Need new coordinates every epoch.
- Change: `data.affine()` + `IconDataset(jitter=...)` / `--jitter`: one random scale (1-j .. 1+j/3) and shift (up to j*GRID, clamped so the target stays on the grid), shared with the exemplars so style stays consistent; flip folded into the same transform. Verified visually on coffee/heart/house/star (scratchpad aug.png).
- Run: 24 epochs, jitter 0.15, flip 0.5, dropout 0.1, p_drop_text 0.2, 4 text tokens, text_scale 19.6.
- Mid-run peek at 12k: worse than v2/v3 (many samples end early in dots/fragments). Val 0.88 is not comparable to earlier runs (positions are now random by design).

## Iter 5: an objective metric, faster sampling, decoding checks
- New `scripts/eval_prompts.py`: 20 common words (heart house star coffee key cloud camera bell lock sun moon search user mail phone trash calendar folder umbrella anchor) x 4 samples in Lucide style; a sample is a hit if its nearest Lucide icon by 48px raster IoU has a name starting with the word. Strict (an open heart counts as a miss), so treat it as relative, not absolute. By eye, v2 is ~40% recognizable when the metric says 16%.
- KV cache: `IconSmith.start_decode` / `step_decode`, tested equal to the full forward (`test_cached_decoding_matches_full_forward`). `generate.sample` now decodes cond + null rows in one batch. Same results, ~6x faster.
- Scores (cfg 2, temp 0.6 unless noted):

| ckpt | hits /80 |
|---|---|
| v2 last | 13 (16%) |
| v2 last, temp 0.3 | 15 (19%) |
| v3 best | 12 (15%) |
| v3 best, temp 0.3 | 12 (15%) |
| v2, CFG on coords only (EOS/Z/M unguided) | 9 (11%) |
| v4 best (jitter 0.15, step 21k) | 0 (0%) |

- Length check (v2, 8 samples/word): CFG 2 makes many icons too short (mail 24-38 tokens vs 135 real; heart ~108 vs 153), which matches the "open heart" / "V-only mail" look. But leaving structure tokens unguided scored worse, so plain CFG stays the default.
- v4 jitter: lost concepts (hearts became circles), and outputs at random scale/offset also break the IoU match. Jitter is dropped. v4 stopped at 21.5k steps. Note: running an eval alongside training filled the 16 GB GPU and stalled training at ~145 s/step for about an hour; evals now run only while the GPU is free.

## Iter 6: canonical subpath order (`runs/v5`)
- Diagnosis: libraries draw subpaths in arbitrary order, so the first tokens (which fix the layout) are needlessly unpredictable; standard SVG generators (DeepSVG, IconShop) sort subpaths.
- Change: `tokenizer.canonical_order` sorts subpaths top-to-bottom then left-to-right *within runs of the same render mode*, so the renderer's merged `<path>` groups (and fill-rule holes) are unchanged. Applied at shard load (`_SetShard(canonical=True)`, `--canonical`, ~40 s extra). Checked on 950 icons across all 23 sets: 540 reordered, worst raster IoU vs original 1.0000. Unit test added (70 tests pass).
- Run: v2 recipe + canonical + p_drop_text 0.2: 10 epochs, flip 0.5, dropout 0.1, 4 text tokens, text_scale 19.6.
- v5 result: best val 0.4781 at step 17,000, final 0.4796 (train 0.30). Prompt eval (Lucide): best ckpt 16/80 (20%), **last ckpt 20/80 (25%), best so far**. By eye about 35-40%: sun 4/4, calendar 4/4, folder 4/4, lock 3/4, mail 3/4, cloud 3/4, user 3/4, house 2/4, bell 2/4. Heart got worse; star/coffee/key/umbrella/anchor still fail.
- Other sets (v5 last): tabler/outline 26/80 (32%), material/rounded 16/80 (20%), phosphor/fill 10/80 (12%). Few sets draw with strokes like Lucide (lucide, tabler, heroicons/ionicons outline), so a stroke-style "star" has only ~4 true examples. Concepts must transfer across fill-outline and stroke drawings.

## Iter 7: bigger model + stronger dropout (`runs/v6`)
- Rationale: v5 val plateaued from step 17k with a train/val gap (0.30 vs 0.48). With little data, a larger model with more regularization usually generalizes better per epoch.
- Memory check: d_model 640 / 10 layers / 10 heads / d_ff 1728 at batch 16 uses ~14.1 GB with no spill (243 W), 25.9k tok/s (v5 config: 32.6k).
- Run: v5 recipe (canonical, flip 0.5, 4 text tokens, text_scale 19.6, p_drop_text 0.2) + the bigger model, dropout 0.2, 12 epochs.
