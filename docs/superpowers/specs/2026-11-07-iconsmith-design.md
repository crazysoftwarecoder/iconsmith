# Set-Conditioned Icon Completion Model

## Context

Goal: a GitHub showcase project in the spirit of FontoGen (a small from-scratch
generative transformer on a structured vector modality), but filling the gaps
FontoGen left: a reproducible data pipeline, released weights, an eval harness,
and a demo.

The frontier open model (OmniSVG, 4B/8B) is a general text/image-to-SVG artist.
It does not take a *style reference*, needs a GPU server, and can emit invalid
SVG. This project builds the thing it structurally can't be: a **set-conditioned
icon completer**. Input: a handful of icons from an existing icon set plus a
name. Output: a new icon in that set's style, guaranteed to be valid SVG.

Decisions already made with the user:

- **Hardware:** everything on this Mac (M5 Pro, 16 cores, PyTorch 2.7 with MPS).
  Target: a full training run in a few hours. No CUDA.
- **Lead feature:** set-conditioned completion from day one. Text-only
  generation falls out of it via conditioning dropout.
- **Eval:** build the held-out set and metrics in v1, score only our model.
  OmniSVG/StarVector baselines are a follow-up.
- Starting directory: `/Users/ashwanthfernando/code/cool-project` (empty, not a
  git repo yet). Python 3.12, `uv` available, cairo installed via brew.

## What gets built

### 1. Data pipeline

**Sources.** Shallow-clone permissively licensed icon repos. A *set* is one
(library, style) pair, e.g. `lucide`, `tabler/outline`, `tabler/filled`,
`phosphor/bold`, `material-symbols/rounded-400`. v1 sources (all MIT, ISC or
Apache; no brand-logo or emoji sets):

Lucide, Tabler, Phosphor, Material Symbols, Heroicons, Iconoir, Feather,
Remix Icon, Bootstrap Icons, Fluent UI System Icons, Ionicons, Octicons.

That is roughly 30 sets and 50k+ icons. `sources.py` is a declarative table:
repo URL, license, glob per style, how to derive the icon name from the path.

**Normalization** (`svg_parse.py`, using `svgelements`): flatten all primitives
(rect, circle, line, polyline, polygon) and transforms to paths; convert arcs
and quadratics to cubics; normalize the viewBox to a unit square; quantize to a
**128x128 grid**. Each icon keeps a set-level *render mode*: `stroke` with a
stroke-width bucket, or `fill`. Outline icons are kept as stroke paths, not
converted to filled outlines, so sequences stay short. Icons that exceed the
max token length (1024) are dropped and counted.

**Tokenization** (`tokenizer.py`): vocab = {PAD, BOS, EOS, M, L, C, Z,
STROKE, FILL_NZ, FILL_EO} + 128 coordinate values. Points are interleaved
scalar tokens (x then y). Every subpath starts with one *per-path mode token*
(stroke, fill nonzero, fill evenodd) followed by M, so an icon is
`BOS (MODE M ... Z)* EOS`. Stroke width stays set-level (a bucket token in the
conditioning prefix). This handles outline sets that contain filled dots,
filled sets that rely on evenodd holes, and libraries authored as filled
outlines, without dropping icons. A grammar table says which tokens may follow
which, used both for a round-trip test and for masking at sampling time so
output is always a valid path.

**Splits.** Two kinds of held-out data, written at build time:

- *in-set test*: 5% of icons from every training set (does it complete a set it
  has seen?)
- *held-out sets*: 3 entire sets never seen in training (does it match a style
  from examples alone?). **A held-out set must come from a library with no
  sibling style in training**, so the number measures true generalization,
  not outline-vs-filled recognition of shapes the model has already seen.
  Candidates: Iconoir, Bootstrap Icons, Octicons (single-style libraries).

**Set granularity.** To keep the 30-way set classification meaningful, near-
duplicate sets are collapsed or excluded: Feather is excluded (Lucide is its
superset), Material Symbols contributes one weight per style family
(outlined/rounded/sharp at weight 400), and only clearly distinct styles of a
library (e.g. `tabler/outline` vs `tabler/filled`) count as separate sets.

**Text conditioning.** Icon names (e.g. `arrow-up-right`) are embedded once in
preprocessing with a frozen `all-MiniLM-L6-v2` sentence encoder (22M params,
runs on MPS/CPU) and stored as a 384-d vector per icon. No text model in the
training loop, mirroring FontoGen.

Output: sharded `.npz`/`.pt` files under `data/processed/` plus a `stats.json`
(icons per set, token-length histogram, drop counts). `data/` is gitignored;
the pipeline is one command.

### 2. Model (`model.py`)

- **Decoder:** decoder-only transformer, pre-norm, RoPE, SwiGLU, PyTorch SDPA
  attention. ~8 layers, d_model 512, 8 heads, ~30M params. Max sequence 1024.
- **Exemplar encoder:** a smaller transformer (4 layers, d 384) sharing the
  token embedding with the decoder. Each of K=8 example icons from the same
  set is encoded and mean-pooled to one style vector.
- **Conditioning prefix** fed to the decoder before BOS:
  `[render-mode token] [text vector → 1 token via linear] [K style tokens]`.
- **Conditioning dropout:** with p=0.1 the style tokens are dropped, with p=0.1
  the text token is dropped. This gives text-only and style-only generation and
  enables classifier-free guidance at sampling time.
- No set-id embedding anywhere, so held-out sets work from exemplars alone.
- Loss: cross-entropy over the icon tokens, padding ignored.

### 3. Training (`train.py`)

- AdamW, cosine schedule with warmup, grad clipping, bf16 autocast on MPS with
  an fp32 fallback flag.
- Exemplars are resampled from the icon's set every step (free augmentation).
- Budget: ~50k icons × ~400 tokens ≈ 20M tokens/epoch. Target 15–25 epochs in a
  few hours on the M5 Pro. Report measured tokens/sec after the first run and
  adjust depth/width to fit.
- Logging: local CSV + matplotlib loss curves (wandb optional via flag).
  Checkpoints to `runs/<name>/`.
- A `--overfit 100` mode trains on 100 icons to confirm loss goes to ~0 before
  any long run.

### 4. Sampling and rendering (`sample.py`, `render.py`)

- Autoregressive sampling with temperature, top-p, grammar masking, and
  optional classifier-free guidance on the style tokens.
- Tokens → SVG path string → SVG file using the set's render mode.
- Raster rendering via `cairosvg` to 64x64 and 256x256 for eval and demo.

### 5. Eval harness (`eval/`)

Run on both held-out splits, reported separately:

- **Validity rate:** parsable SVG, closed paths where required (expected 100%
  with masking; reported anyway).
- **Reconstruction vs. ground truth:** raster IoU at 64x64 and Chamfer distance
  on sampled contour points, for the held-out icon given its real name and
  exemplars from its set.
- **Style consistency:** CLIP (`open_clip` ViT-B/32) image embeddings of
  generated icons; nearest-centroid *set classification accuracy* against all
  sets, plus mean cosine to the true set's centroid. Also cheap geometric stats
  (path count, point count) vs. the set's distribution.
- **Semantic match:** CLIP text-image score between the icon name and the
  render. Weak for icons, but reported.
- Output: `eval/results.md` table plus a sample grid image per set.

### 6. Demo and docs

- `app.py`: Gradio app. Pick a set from the dataset or upload 4–8 SVGs, type a
  name, get a grid of 8 candidates, download any as SVG.
- README: one hero figure (sample grid above the fold), loss curve, eval table,
  how to retrain on your own icon set in one command, a "what surprised me"
  section. Model weights uploaded to Hugging Face at the end.

## Repo layout

```
pyproject.toml            # uv-managed, python 3.12
README.md
src/iconsmith/            # repo and package name: iconsmith (user's choice)
  sources.py  fetch.py  svg_parse.py  tokenizer.py  dataset.py
  model.py  train.py  sample.py  render.py  app.py
  eval/metrics.py  eval/run_eval.py
tests/                    # pytest: tokenizer round-trip, grammar mask, parse edge cases
scripts/build_dataset.sh  scripts/train.sh
docs/                     # writeup drafts, figures
data/  runs/              # gitignored
```

Dependencies: torch, svgelements, cairosvg, sentence-transformers, open_clip_torch,
numpy, gradio, matplotlib, pytest, ruff.

## Milestones (each ends in something runnable)

1. **Repo + data pipeline.** `git init`, uv project, fetch 12 sources, parse,
   tokenize, build splits. Tests: tokenizer round-trip is lossless; rendered
   round-trip of 500 random icons against the original SVG render, at 128x128,
   has **mean IoU > 0.98 and 5th-percentile IoU > 0.90**. If the percentile
   fails on thin-stroke sets, move to a 256 grid before building the model.
   Deliverable: `stats.json` and a contact sheet of parsed icons per set.
2. **Model + overfit.** Model builds on MPS, 100-icon overfit reaches near-zero
   loss, measured tokens/sec on the M5 Pro recorded. Decide final depth/width.
3. **Full training run.** Multi-hour run, loss curves, checkpoint. Sample grids
   for 5 sets including one held-out set.
4. **Eval harness.** All metrics above on both held-out splits, results table.
5. **Demo + README + weights.** Gradio app, hero figure, writeup, HF upload.

## Verification

- `pytest` passes (tokenizer, grammar, parser).
- Round-trip IoU check from milestone 1 passes.
- Overfit run reaches loss < 0.05.
- Full run: validation loss decreases and plateaus; sample grids are visually
  coherent per set.
- Eval table produced for both splits; set-classification accuracy on
  *held-out sets* is well above chance (1/30), which is the headline number.
- `python -m iconsmith.app` serves and generates an icon end to end.

## Calls I'm making (override any of these)

- 128-grid quantization and interleaved x/y scalar tokens (simplest, lossless
  enough at icon scale).
- Stroke icons stay as strokes rather than being converted to filled outlines.
  Render mode is a per-subpath token (stroke / fill nonzero / fill evenodd);
  stroke width is set-level.
- Held-out sets come only from libraries with no sibling style in training;
  Feather dropped and Material limited to one weight per style, so the
  classification metric isn't inflated by near-duplicate sets.
- Round-trip acceptance is mean IoU > 0.98 and 5th percentile > 0.90 at
  128x128, with a fallback to a 256 grid if thin strokes fail.
- Frozen MiniLM for names rather than training a text encoder.
- K=8 exemplars, mean-pooled.
- CLIP ViT-B/32 as the style/semantic judge for eval.
- Only MIT/ISC/Apache sources in v1; CC-BY sets (Font Awesome) can be added
  with attribution later.
- Three whole sets held out, picked during milestone 1 once set sizes are known.

## Out of scope for v1

ONNX/browser export, OmniSVG/StarVector baselines, illustrations or
multi-colour SVGs, flow-matching decoder, Figma plugin.
