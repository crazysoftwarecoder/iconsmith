# iconsmith — project context for Claude Code

A small (33.6M param) decoder-only transformer that draws a new icon in the style of an
existing icon set. Input: K=8 exemplar icons from a set + an icon name. Output: path tokens
decoded under a grammar mask, so every sample is a valid SVG.

Read these before changing anything substantial:

- Design spec: `docs/superpowers/specs/2026-11-07-iconsmith-design.md` (the authority)
- Implementation plan (milestones 1–2): `docs/superpowers/plans/2026-11-07-iconsmith-foundation.md`
- Progress ledger with every ruling made during implementation:
  `docs/superpowers/plans/2026-11-07-iconsmith-foundation-ledger.md` (a committed copy of the
  git-ignored `.superpowers/sdd/.../progress.md`; append new rulings to both)

## Hardware

- Development/smoke tests: any machine (MPS or CPU is fine for `pytest`).
- **Training runs go on the RTX 4080 (16 GB) box, not the Mac.** `resolve_device("auto")`
  picks `cuda` > `mps` > `cpu`. The default config (d_model 512, 8 layers, batch 32, bf16)
  fits the 4080 comfortably. On MPS an earlier version OOMed at 58 GiB because explicit
  SDPA masks forced the math kernel; the decoder now uses `is_causal` and exemplars are
  truncated to 256 tokens (`EXEMPLAR_MAX_LEN`), so memory is modest everywhere.

## Setting up on a fresh machine (e.g. the 4080 box)

```bash
git clone <this repo> iconsmith && cd iconsmith
uv sync                       # Python 3.12; installs torch (CUDA wheels on Linux/Windows)
uv run pytest -q              # 68 tests, ~10 s; first run downloads MiniLM (90 MB)
scripts/build_dataset.sh      # clones 9 icon repos + 2 npm tarballs (~540 MB) into data/raw,
                              # then builds data/processed (~3 min). data/ is git-ignored.
uv run python scripts/check_roundtrip.py data/raw 500 256   # expect ACCEPT (mean>0.98, p5>0.90)
```

If torch installs without CUDA on the 4080 box, pin the CUDA index:
`uv pip install torch --index-url https://download.pytorch.org/whl/cu124` (or edit
`pyproject.toml` with a `[tool.uv.sources]` torch entry) and re-run `uv sync`.

## Running training

```bash
# 1. Overfit sanity check (must reach train loss < 0.05):
uv run python -m iconsmith.train --processed data/processed --run-dir runs/overfit100 \
    --overfit 100 --epochs 200 --batch-size 16 --warmup-steps 50 --log-every 50 --eval-every 200

# 2. Throughput measurement (record tokens/sec from the last log line):
uv run python -m iconsmith.train --processed data/processed --run-dir runs/tput \
    --max-steps 200 --batch-size 32 --eval-every 200

# 3. Full run (20 epochs; ~10M train tokens/epoch):
scripts/train.sh v1
```

Each run writes `runs/<name>/log.csv`, `loss.png`, `ckpt_last.pt`, `config.json`, and shows
a tqdm progress bar over total steps with live loss and tokens/sec. Useful flags:
`--no-bf16`, `--device cuda:0`, `--d-model/--n-layers/--n-heads/--d-ff`, `--max-steps`.

## Status (as of 2026-11-07)

Plan tasks 1–9 complete and committed. Task 10 (trainer) code is complete and tested; its
last two verification steps still need a GPU run:

- [ ] Step 5: overfit run reaches train loss < 0.05 (`runs/overfit100`)
- [ ] Step 6: measure tokens/sec; target ≤ 20 min/epoch. If over, drop `--n-layers 6`
      and/or `--d-model 384`; record the chosen config under "Measured" at the bottom of
      the plan file and in `scripts/train.sh`.

Then milestones 3–5 (full training run, eval harness, Gradio demo + README + HF weights)
get a second plan via the `superpowers:writing-plans` skill.

Dataset facts (GRID=256): 23 sets from 11 libraries, 40,100 icons parsed, 243 dropped as
too long (>1024 tokens), 0 parse errors, ~10M training tokens/epoch. Held-out libraries
(never trained on): iconoir (regular+solid), bootstrap, octicons. In-set test split: 5%.

## Layout

```
src/iconsmith/
  tokenizer.py     GRID=256, vocab (PAD BOS EOS M L C Z STROKE FILL_NZ FILL_EO + coords),
                   encode/decode, GrammarState for constrained sampling
  svg_parse.py     SVG -> quantised subpaths (svgelements); per-subpath fill/stroke mode
  render.py        Icon -> SVG text; rasterize via resvg-py; IoU
  sources.py       declarative table of libraries/sets/licenses/globs
  fetch.py         git clone / npm tarball into data/raw
  dataset_build.py per-set .npz shards + sets.json + stats.json; MiniLM name embeddings
  text_embed.py    frozen all-MiniLM-L6-v2 (preprocessing only)
  data.py          IconDataset (train | inset_test | heldout), same-set exemplar sampling
  model.py         IconSmith: exemplar encoder + RoPE/SwiGLU decoder, conditioning dropout
  train.py         AdamW, warmup+cosine, bf16 autocast, CSV log, loss.png, checkpoints
scripts/           build_dataset.sh, train.sh, check_roundtrip.py, contact_sheet.py
tests/             pytest; fixtures in tests/fixtures
docs/figures/      roundtrip.json, contact_<set>.png
```

## Conventions

- TDD: write the failing test first, watch it fail, then implement. `uv run pytest -q`
  and `uv run ruff check .` must be clean before every commit.
- Commit per task; messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Deviations from the plan are recorded as `Ruling:` lines in the ledger, never silent.
- Only MIT/ISC/Apache icon sources. No brand-logo or emoji sets.

## Key design decisions (see spec and ledger for the why)

- `Z` is the mandatory end-of-subpath token; a closed stroke is an explicit `L` back to its
  start. Zero-length strokes (Tabler/Lucide dots) are kept as `M p L p Z`.
- Render mode is per subpath (stroke / fill nonzero / fill evenodd); stroke width is a
  set-level bucket token in the conditioning prefix.
- The renderer merges consecutive same-mode subpaths into one `<path>` so fill-rule holes
  survive. Known limitation: separate opposite-wound overlapping source elements gain a hole.
- Coordinate frame is `(0, 0, width, height)` as reified by svgelements (it applies the
  viewBox transform itself). Material Symbols' `viewBox="0 -960 960 960"` depends on this.
- Rasterizer is resvg-py (self-contained wheel); cairosvg could not find Homebrew cairo.
