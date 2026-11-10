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
- v6 result: best val 0.4524 at 19k (v5: 0.4781) but overfit by the end (train 0.13, val 0.488). Prompt eval: last 17/80 (21%), best 18/80 (22%), i.e. no better than v5 (20-25%). A bigger model lowers loss but does not make names more recognizable.
- Best-of-N reranking (`--best-of`, keep the candidate with the highest summed log p(x|name) - log p(x|no name)): v6 21/80 (26%) but v5 10/80 (12%). The sum favors long, overloaded scribbles (more tokens, more score). Off by default.
- With 4 samples per word the metric moves +/-5 points between seeds, so differences under ~8 points are noise.

## Iter 8: retrieval of content exemplars (diagnosis of the plateau)
- Across v2-v6 the visual hit rate plateaus at ~35-40% whatever the model size, schedule or decoding. Learning name -> shape from scratch is too hard at this data size: a stroke-style "star" has ~4 true training examples, and the model must also transfer shapes from fill-outline sets.
- But nearly every prompt name already exists in *other* sets (14,376 unique names; `heart` x15, `coffee` x8; 86% of held-out names occur elsewhere). So: give the model C content exemplars, the icons whose names are nearest to the prompt (MiniLM cosine) from *other* sets. The task becomes "redraw this known shape in the target style", which is much easier, and it still works for novel names via nearest-name retrieval.
- Design: content exemplars share the exemplar encoder, get prefix type 3, and are flipped with the target; dropped together with the name (CFG then guides toward both), plus independent content dropout so the model still works without them. Never retrieved from the target's own set, and only from training rows.
- Implementation: `data.ContentPool` (training rows of all training sets, nearest names by MiniLM cosine, *excluding the target's whole library*, because sibling sets like tabler/outline vs tabler/filled share geometry and would teach copying). `IconDataset(n_content=2)` samples 2 of the top 4 neighbours each time and flips them with the target. Model: content icons go through the shared exemplar encoder, then 16 learned-query attention slots each (a mean would lose the layout), prefix type 3, dropped with the name and independently (p 0.2). `generate.py` / `eval_prompts.py` retrieve the nearest 2 at inference and print what they draw from. Tests: content prefix length, all-pad content row stays finite, cached decoding equal to the full forward.
- Retrieval check: lucide `coffee` -> phosphor/bold, material/sharp, material/rounded, phosphor/regular coffee; `heart` -> phosphor/thin, phosphor/fill, remix/line, phosphor/bold heart; `star` -> ionicons/outline, material/outlined, remix/line, material/rounded star.
- Run v7: v5 recipe (34M, canonical, flip 0.5, dropout 0.1, p_drop_text 0.2) + `--n-content 2`, 10 epochs, 26.7k tok/s, 11.9 GB.
- v7 result: final val 0.4708 (v5 0.4796). Prompt eval 16/80 (20%), visually about the same as v5 (house, lock, search, user, mail, phone, calendar good; coffee and heart still fail).
- Diagnosis (`scratchpad/diag_content.py`, inset_test, 40 batches): full 0.4709 vs **no content 0.4715**, no text but keep content 0.5004, neither 0.5020. **The model ignores the content exemplars.** 16 pooled slots per icon compress the shape too much for the decoder to copy from. Also, retrieved content was often fill-outline icons (phosphor/material) for a stroke target.

## Iter 9: content as full token sequences + same-render preference (`runs/v8`)
- Change: `ModelConfig.content_mode = "tokens"`: every encoded content token (shared exemplar encoder, then `content_out`) goes into the decoder prefix, with pads replaced by one learned constant; the prefix length is now dynamic (`forward` slices at `prefix.shape[1]`). "slots" mode is kept so v7 loads. Test: cached decoding equals the full forward in tokens mode (72 tests pass).
- Retrieval: icons drawn the same way as the target (stroke vs fill) get +0.15 cosine (`SAME_RENDER_BONUS`). Lucide now retrieves `coffee` -> tabler/outline coffee; `coffee cup` -> tabler coffee, tabler cup; `heart` -> tabler, heroicons outline; `star` -> ionicons, heroicons, tabler outline; `key`/`umbrella`/`anchor`/`rocket` -> tabler/ionicons outline.
- Run v8: v5 recipe + `--n-content 1 --content-mode tokens`, 10 epochs, 24k tok/s, 15.0 GB at 252 W (no spill). Early sign: train loss ~0.8-0.96 at step 2.7k, lower than earlier runs at that point.
- v8 result: best val 0.4511 at 17k, final 0.456 (v5 0.4796, v7 0.4708). Content is now used heavily. Prompt eval: cfg 2 16/80 (20%), cfg 1.5 15/80 (19%), **cfg 1 5/80 (6%)**. No improvement, and guidance off is terrible even with the exact Tabler icon as content.

## Iter 10: the real sampling failure, arbitrary start points
- Check (`scratchpad/infer_check.py`, `cache_check.py`) on Lucide *training* icons with generate's exact conditioning: teacher-forced loss is low (coffee 0.30, star 0.27; 92% next-token accuracy), and the cached decoder equals the full forward (fp32 max diff 1e-5; bf16 argmax agreement 99.6%). Yet near-greedy samples (temp 1e-4, cfg 1) are garbage: every star stops at 34 tokens.
- Token trace for Lucide `star`:
  - real:   `BOS STROKE M 132 34 C 133 34 ...` (starts at the top point)
  - greedy: `BOS STROKE M 122 79 L 196 122 Z STROKE M 108 167 L ...` (a loop of tiny 2-point strokes)
  - Each subpath's start vertex is arbitrary in the sources (a star can start at any of its 10 vertices). x and y are separate tokens, so the sampler takes x from one plausible start and y from another: (122, 79) is on no star vertex. From an off-icon point the model is lost. Teacher forcing hides this because it is always handed the real start. **This affects every model so far**, and explains why sampled icons look much worse than the loss suggests (open hearts, fragments, early stops).
- Fix: `tokenizer.canonical_start`: closed subpaths (last vertex == first) are rotated to start at their top-left vertex (min y, then x); open strokes are reversed when their far end is more top-left; open fills are left alone (reversal flips winding, changing nonzero holes). `canonical_order(starts=True)` then sorts by start point. Exposed as canonical level 2 (`--canonical 2`; level 1 = order only, so older checkpoints load as trained). Checked on 1,443 icons across all sets: 1,346 changed, no length changes, worst raster IoU vs original 0.9992 (sub-pixel join difference at the rotated start). Unit test added (73 tests pass).
- Run v9: v8 recipe with `--canonical 2`.
- Caught before training: flip augmentation mirrors top-left starts to top-right, so half the samples would break the convention again. `data.recanonical` now re-applies the canonical level to the target and content after augmentation (content is fetched untruncated, re-canonicalized, then cut to 384 tokens, because a truncated icon has no EOS to decode). ~2.4 ms/item. Exemplars are not re-canonicalized (style only).
- v9 result: best val **0.4364** at 19k, final 0.4376 (v8 0.4511), the lowest so far. Prompt eval (Lucide, 4 samples/word): cfg 1 17/80 (21%), cfg 1.5 17/80, cfg 2 16/80, cfg 3 14/80. Guidance-off went from 5/80 (v8) to 17/80, so canonical starts made the conditional model itself much better. Sun, calendar, search, folder, user, mail and cloud are clean; heart, star, house, coffee and key are still garbage.

## Iter 11: why heart/star/coffee still fail, start point plus content ignored
- Greedy on Lucide *training* icons (`scratchpad/diverge.py`): heart gives a 30-token triangle (real 153 tokens), coffee 34 tokens of bars (real 172). Teacher-forced, every token after the first point is predicted well, but **p(real first x) = 0.00 for heart, 0.09 for coffee**. The model commits to a wrong start (heart x=186 vs 72) and draws something else from there.
- First-x distribution given the prompt (`scratchpad/heartflip.py`): sun (p 0.65 on the right x) and calendar (0.34) are peaked, which is exactly the prompts that work. Heart, star and house are nearly flat (max p ~0.1). Flip is *not* the cause: the canonical start of flipped heart/star/house equals the original.
- Per-position loss on inset_test (`scratchpad/posloss.py`): the name lowers loss on the first coordinate from 2.0 to 1.29 nats and on tokens 3-8 from 1.23 to 0.92, but after token 32 it barely matters (0.44 vs 0.46). So the name only steers the start; the rest is local continuation.
- **Content is completely ignored** (`scratchpad/posloss2.py`): content rolled to a *different* icon in the batch gives identical loss at every position (first coord 1.290 vs 1.286). The retrieved Heroicons heart starts at (82, 40), right next to Lucide's (72, 43), yet the model puts p 0.11 on x=186. (Correction to iter 9: v8's lower loss came from token mode and canonical order, not from reading content.) Cross-library neighbours rarely line up with the target exactly, so the gradient for copying is weak and the model learned to skip the channel.
- Fix: teach copying directly. `IconDataset(p_self_content=0.5)`: half of the time the content exemplar is a *warped copy of the (augmented) target*: random scale 0.9-1.05, shift about +/-13 units, and a smooth per-axis wobble of ~4 grid units (`data.warp`, a lookup on coordinate value, so coincident points such as closed subpaths stay coincident), then re-canonicalized. Visually these look like the same icon redrawn by hand. The other half stays a real cross-library neighbour, so the task is "redraw this shape in the target style". At inference the content is the nearest same-name icon from another library. Test added (74 pass).
- Run v10: v9 recipe + `--p-self-content 0.5`.
- v10 result: best val 0.4516 (worse than v9 0.4364; val content is real cross-library neighbours). Prompt eval **1/80 at cfg 1, 2/80 at cfg 2**: much worse. Per-position loss: the name gap shrank (first coord 2.04 no-name vs 1.44 with name; v9 2.00 vs 1.29) and shuffled cross-library content still makes almost no difference (1.463 vs 1.444). The model learned to copy only *token-aligned* self-copies; a real neighbour (different commands and point counts) is out of distribution, and whenever the self-copy gave the answer the name was not needed, so the name signal weakened. **Reverted; v9 stays the base.**

## Iter 12: inference-time fixes on v9 (reference rerank, forced start)
- Idea: v9 continues a drawing well from the right start, and the retrieved same-name icon from another library (never the target's own library) is a good picture of the target. Use it at sampling time rather than as model input.
- `generate.soft_raster` (32 px alpha, Gaussian blur, unit norm) and `sample_best(ref_svg=...)`: draw best_of candidates and keep the one most similar to the reference silhouette (`--rerank-ref`). `reference_svgs` returns (description, svg, start prefix) per prompt.
- `sample(prefix=...)` forces tokens after BOS (None = sampled); `--start-ref` forces the reference's canonical first point `[free mode, M, x, y]`.
- Results (Lucide, cfg 1.5, temp 0.8, 4 samples/word):
  - rerank-ref, best-of 8: 21/80 (26%). Lock, sun, search, calendar, folder clean; heart/star/coffee/key never drawn, so nothing to pick.
  - start-ref + rerank, best-of 8: 10/80, but **heart 4/4 perfect** for the first time. Other words broke where the reference library starts the icon elsewhere than this set would (search, sun).
  - **mixed** (8 free + 8 forced candidates, reranked together): **23/80 (29%), best so far**. By eye about 45%: heart, sun, search, mail, calendar 4/4; folder, lock mostly; user, bell, house partly. Coffee, star, key, camera, moon, phone, umbrella, anchor still fail.

## Iter 13: more data
- Diagnosis behind it: v6 (bigger model) lowered loss but not recognisability, the train/val gap is large (0.25 vs 0.44), most names have only a handful of examples, and the name only steers the first ~30 tokens. All point to data, not capacity.
- Fetched and added (licence checked, MIT/Apache only): IconPark (Apache-2.0), MingCute line/fill (Apache-2.0), Eva outline/fill (MIT), Akar (MIT), Majesticons line/solid (MIT), Radix (MIT), Carbon (Apache-2.0). Rejected: css.gg (custom licence since 2.1.2), Material Design Icons (Pictogrammers Free License), Teenyicons (repo gone). Fluent was already in.
- Code: `SetSpec.exclude_re` (brand logos / emoji / crypto folders, Carbon `logo--`, Radix `-logo`, Akar `-fill`, a brand-name regex); `Source.fix_svg` (IconPark two-tone to its outline theme: blue inner fills -> none, white inner strokes -> black); the parser drops `opacity:0` / `fill-opacity:0` elements (every Eva icon has a hidden full-size rect that parsed as a black square). New sources are appended so the shared split RNG leaves the old sets' splits unchanged. Tests added (77 pass). Built into `data/processed_v2` so v9 still evaluates on the old data.
- Build (`data/processed_v2`): **51,466 icons** (was 40,100, +28%), 0 parse errors, 276 too long; old shards byte-identical. Stroke-style icons, the ones closest to Lucide: +3.2k (IconPark 2,536, Akar 335, Majesticons line 380), about +40%. `check_roundtrip.py` now applies `fix_svg` too: 2,000 icons at 256 px, ACCEPT, new sets 0.974-0.990 mean IoU (old sets 0.962-0.993). Random contact sheet of the new sets looks right (IconPark as clean outlines, no Eva squares, no logos).
- Run v11: v9 recipe on `data/processed_v2` (45,055 train / 2,371 inset_test, 28,160 steps, ~100 min).
- v11 result: best val 0.4069 at 25k (new val set incl. new libraries, so not comparable to v9's 0.4364). Survived a CUDA OOM at step 37 on the first launch (near the 16 GB limit: 15.98 GB, 14k tok/s vs 21k for v9), relaunched unchanged. Prompt eval (Lucide): plain cfg 1.5 18/80 (22%); mixed rerank 22/80 (28%), but by eye clearly better than v9: star 2-3/4, bell 3-4/4, camera 2/4 and trash 2-3/4 now work, lock 4/4 (the metric scores it 0), heart, sun, mail, calendar, user good. ~33/80 by eye. Coffee still fails: "coffee cup", "coffee", "mug", "cup of tea" all come out as boxes and bars.

## Iter 14: why coffee fails, local decisions without a global plan
- Forcing a prefix of the *real* Lucide icon (training rows) and decoding (`scratchpad/forcecheck.py`, `divpos.py`): even with the first 60 real tokens forced, star never completes (greedy loops past 1024 tokens), coffee becomes a jar with no handle, key never closes.
- Teacher-forced argmax is wrong at 16-22 positions per icon (8%). Greedy from 60 forced tokens diverges exactly there: coffee at token 75 (real x 74, greedy 117, a narrower cup, then self-consistent and handle-less); key at token 92 (real ends the shaft with `L 21 223 Z`, greedy keeps curving). These are *decision points* where several continuations are locally plausible; the model has no global picture of the icon to choose between them. Teacher forcing hides it by always supplying the right branch. More data/model size only makes each local step more plausible.
- Fix: give the decoder a global plan it can actually use, a **picture**. Token content from another library failed because its tokens do not line up with the target's (iter 9-11), but a raster does: pixel (x, y) is coordinate (x, y) however the paths were written.
- `ModelConfig.content_mode = "raster"` (`raster_px` 32, `raster_patch` 4): the content icon's 32x32 coverage map is cut into 64 patch tokens (MLP on 4x4 patches + learned 2-D position), prefix type 3; null picture has the same 64-token length so CFG/dropout never shift positions. `render.alpha` (anti-aliased coverage), `data.content_picture` (drawn at stroke 0.083 whatever the source set). `IconDataset(content_raster=32)`: half the time a picture of the warped (`data.warp`, ~4 units) target, half the time a same-name icon from another library; val uses real neighbours only. `generate.retrieve_content` renders the reference at inference. Tests: fixed prefix length, the picture changes the output, cached decoding equals full forward, dataset pictures in [0,1] and collated (79 pass). Visual check of 12 training items: self pictures near-identical, neighbour pictures the same concept drawn differently.
- Run v12: v11 recipe + `--content-mode raster --raster-px 32 --p-self-content 0.5`. 11 GB (was 16), 5.6 steps/s, ~85 min. v11 val at 3k/5k for comparison: 3000,0.7534 5000,0.6380 

- v12 probes (`scratchpad/posloss_self.py` with every content a warped self-picture; `posloss2.py` with pictures shuffled across the batch): at 6k and 19k steps the loss with a shuffled picture equals the loss with the right one, **even for self-pictures**. `raster_in` weights barely moved from init. The model never learned to map coordinate token values to patch positions: a token embedding for x=74 has nothing in common with "column 9 of the picture". Stopped at ~19.2k (val 0.4521 vs v11 0.4269 at the same step).
- Fix attempt: `ModelConfig.coord_fourier` (16): sin/cos features of the coordinate value (16 log-spaced frequencies, 1 to GRID/4 cycles) are added to every coordinate token embedding (`model.embed`), and the *same* features of each patch centre are added to the raster patch tokens. Token "x=74" and patch "column 9" now share an explicit positional code, so attention can match them. Tests: cached decoding equals full forward; only coordinate tokens get features (80 pass).
- Run v13: v12 recipe + `--coord-fourier 16`. Finished 28,160 steps, best val 0.4219 (v11 0.4069 on the same val).
- v13 probe at 28k: self-picture full 1.574 vs shuffled 1.602 on the first coordinate, and identical after token 8 (0.617 vs 0.618, 0.458 vs 0.459). **Still ignores the picture.** Dropping the raster-plan line of work; v11 + mixed rerank (iter 12) is the current best.
- Why it plausibly fails: next-token loss is already low from local continuation (~0.45 nats after token 32), so the gradient that would teach the model to read the picture is tiny compared with what it gets from the previous tokens. The ~8% decision-point tokens are where the picture would help, and they're too few to drive learning. Options left: a model that brings a prior (fine-tuning a pretrained code LM to write paths), or a hierarchical plan-then-draw model (DeepSVG-style), where the plan is part of the architecture rather than an optional input.
