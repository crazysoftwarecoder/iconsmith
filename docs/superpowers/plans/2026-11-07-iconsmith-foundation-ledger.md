# SDD ledger — plan: docs/superpowers/plans/2026-11-07-iconsmith-foundation.md
Spec: docs/superpowers/specs/2026-11-07-iconsmith-design.md (read)
Pre-flight (shared interfaces):
- T2→T3/T4/T6: Icon/Subpath/GRID/encode/decode — consistent.
- T3→T4/T6/T7: ParsedIcon(icon, stroke_width_rel) — consistent.
- T5→T6/T7: iter_set_files / SOURCES monkeypatch — consistent.
- T6→T8: npz layout (tokens/offsets/names/text_emb/split) + sets.json — consistent.
- T8→T9/T10: batch dict keys tokens/text_emb/exemplars/stroke_bucket — consistent with model.loss(batch).
Setup: Ruling: work directly on `main` — brand-new repo with no other history, user said "let's start"; cost if wrong: none (no shared branch exists).
Task 1: Ruling: rasterizer is resvg-py, not cairosvg — cairocffi cannot dlopen Homebrew's libcairo on this Mac without DYLD env hacks; resvg-py is a self-contained wheel (verified: renders a square, 0.25 coverage). Task 4 uses `resvg_py.svg_to_bytes(svg_string=..., width=px, height=px)`. Cost if wrong: swap one function in render.py.
Task 1: complete (commits 4c5d660..4c5d660, tests: uv run pytest -q → 1 passed in 0.00s)
Task 2: Ruling: Z is the mandatory end-of-subpath token (grammar: EOS/MODE only after Z). A closed stroke is encoded as an explicit L back to its first point before Z; the renderer never emits 'Z' for stroke subpaths. With round caps/joins this renders identically. Cost if wrong: a visible join artifact on closed strokes with non-round joins (none of the v1 stroke sets use them).
Task 2: complete (commits 4c5d660..d19b55d, tests: uv run pytest tests/test_tokenizer.py tests/test_smoke.py -q → 12 passed in 0.06s)
Task 3: Ruling: fill subpaths with fewer than 3 distinct points are dropped (zero area, e.g. a <line> whose default fill is black); stroke subpaths need 2. Cost if wrong: none visible.
Task 3: complete (commits d19b55d..3dabc5f, tests: uv run pytest tests/test_svg_parse.py tests/test_tokenizer.py tests/test_smoke.py -q → 25 passed in 0.05s)
Task 4: Ruling: renderer merges consecutive same-mode subpaths into one <path> so evenodd/nonzero holes survive (the token stream has no element boundaries). Cost if wrong: two separate overlapping evenodd elements in a source icon would gain a hole on round-trip; rare in icon sets, caught by the Task 7 IoU check.
Task 4: complete (commits 3dabc5f..6c87e5b, tests: uv run pytest -q → 32 passed in 0.06s)
Task 5: fetch verified: 23 sets, all >200 files (min heroicons 324, max tabler/outline 5203, total 44347). Material from npm @material-symbols/svg-400 0.48.3, Fluent from npm @fluentui/svg-icons latest.
Task 5: complete (commits 6c87e5b..cfcd578, tests: uv run pytest tests/test_sources.py -q → 7 passed in 0.01s)
Task 7: Ruling: GRID = 256 (spec allowed the fallback). At 128 the round-trip mean was 0.961 (<0.98) across all sets, not just thin strokes; at 256 it is 0.980 / p5 0.957 → ACCEPT. Cost if wrong: coordinate vocab 266 instead of 138; negligible.
Task 7: Ruling: known limitation kept — consecutive fill_nz subpaths from *separate* source elements are merged into one <path>, so opposite-wound overlaps become holes (e.g. bootstrap heart-pulse-fill, IoU 0.85). Rare (bootstrap set mean 0.985); the alternative breaks the far more common single-path nonzero-hole idiom. Cost if wrong: a few icons per thousand render with a spurious hole.
Task 3 (late): Ruling: zero-length stroke subpaths are kept as M p L p Z (dots in Tabler/Lucide); fills still need 3 distinct points.
Task 3 (late): Ruling: coordinate frame is (0,0,width,height) as reified by svgelements, not the raw viewBox — fixes Material (was IoU 0.0).
Task 8: Ruling: exemplars for a training set are drawn only from its split==0 rows (never from in-set test rows), so the in-set test icons are never seen as exemplars either. Cost if wrong: none.
Task 8: complete (commits 6abd9b1..e9e089a, tests: uv run pytest tests/test_data.py -q → 7 passed in 0.75s)
Task 9: Ruling: the exemplar encoder shares tok_emb through a d_model->enc_d_model projection (widths differ); its FFN width is 8/3*enc_d rounded to 64. Prefix slots get a learned type embedding (stroke/text/style). Cost if wrong: none functional.
Task 9: complete (commits e9e089a..313fc46, tests: uv run pytest tests/test_model.py -q → 8 passed in 0.58s)
Task 6: full build (GRID=256): 40,343 files, 40,100 parsed, 243 too long (max 3.2% bootstrap), 0 empty, 0 parse errors, 11.3M tokens total (~10M train/epoch). p50 token length 90 (iconoir/regular) .. 447 (phosphor/thin). 23 sets; 4 held-out sets from 3 libraries.
Task 7: contact sheets eyeballed (lucide stroke, material/rounded fill, octicons held-out): match originals. roundtrip.json ACCEPT (mean 0.980, p5 0.957, 500 icons @256px).
Task 6: complete (commits cfcd578..9139987, tests: uv run pytest tests/test_dataset_build.py tests/test_text_embed.py -q → 5 passed in 6.43s)
Task 7: complete (commits 5bd4d71..9139987, tests: uv run pytest tests/test_render.py tests/test_svg_parse.py -q → 25 passed in 0.07s)
Task 10: Ruling: exemplars truncated to 256 tokens (EXEMPLAR_MAX_LEN) and decoder uses is_causal fused SDPA without a padding mask — fixes MPS OOM (58 GiB) from materialised S x S masks. Cost if wrong: long exemplars lose their tail (style is in the first subpaths); test_trailing_padding_does_not_change_earlier_logits guards the mask removal.
