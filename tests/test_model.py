import torch

from iconsmith.model import IconSmith, ModelConfig
from iconsmith.tokenizer import BOS, COORD_OFFSET, EOS, PAD, STROKE, VOCAB_SIZE, M, Z


def tiny():
    return ModelConfig(
        d_model=64, n_layers=2, n_heads=4, d_ff=128,
        enc_d_model=32, enc_layers=1, enc_heads=4, max_len=64,
    )


def fake_batch(B=2, L=12, k=8, Le=10):
    tok = torch.full((B, L), PAD)
    tok[:, 0] = BOS
    tok[:, 1] = STROKE
    tok[:, 2] = M
    tok[:, 3:5] = COORD_OFFSET + 3
    tok[:, 5] = Z
    tok[:, 6] = EOS
    ex = torch.full((B, k, Le), PAD)
    ex[:, :, :7] = tok[:, None, :7]
    return dict(
        tokens=tok, text_emb=torch.randn(B, 384), exemplars=ex,
        stroke_bucket=torch.tensor([1, 0]),
    )


def test_forward_shape():
    m = IconSmith(tiny()).eval()
    b = fake_batch()
    out = m(b["tokens"], b["text_emb"], b["exemplars"], b["stroke_bucket"])
    assert out.shape == (2, 11, VOCAB_SIZE)


def test_loss_finite_and_backward():
    m = IconSmith(tiny()).train()
    b = fake_batch()
    loss = m.loss(b)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.tok_emb.weight.grad is not None


def test_causal_no_future_leak():
    m = IconSmith(tiny()).eval()
    b = fake_batch()
    out1 = m(b["tokens"], b["text_emb"], b["exemplars"], b["stroke_bucket"])
    b["tokens"][:, 6] = COORD_OFFSET + 9  # change a later token
    out2 = m(b["tokens"], b["text_emb"], b["exemplars"], b["stroke_bucket"])
    assert torch.allclose(out1[:, :5], out2[:, :5], atol=1e-5)


def test_style_dropout_changes_output():
    m = IconSmith(tiny()).eval()
    b = fake_batch()
    a = m(b["tokens"], b["text_emb"], b["exemplars"], b["stroke_bucket"])
    c = m(
        b["tokens"], b["text_emb"], b["exemplars"], b["stroke_bucket"],
        drop_style=torch.tensor([True, True]),
    )
    assert not torch.allclose(a, c)


def test_text_dropout_changes_output():
    m = IconSmith(tiny()).eval()
    b = fake_batch()
    a = m(b["tokens"], b["text_emb"], b["exemplars"], b["stroke_bucket"])
    c = m(
        b["tokens"], b["text_emb"], b["exemplars"], b["stroke_bucket"],
        drop_text=torch.tensor([True, True]),
    )
    assert not torch.allclose(a, c)


def test_exemplar_padding_is_masked():
    m = IconSmith(tiny()).eval()
    b = fake_batch(Le=10)
    a = m(b["tokens"], b["text_emb"], b["exemplars"], b["stroke_bucket"])
    ex_longer = torch.full((2, 8, 20), PAD)
    ex_longer[:, :, :10] = b["exemplars"]
    c = m(b["tokens"], b["text_emb"], ex_longer, b["stroke_bucket"])
    assert torch.allclose(a, c, atol=1e-5)


def test_param_count_default_config():
    n = IconSmith(ModelConfig()).n_params()
    assert 25_000_000 < n < 45_000_000


def test_runs_on_mps_if_available():
    if not torch.backends.mps.is_available():
        return
    m = IconSmith(tiny()).to("mps")
    b = {k: (v.to("mps") if torch.is_tensor(v) else v) for k, v in fake_batch().items()}
    assert torch.isfinite(m.loss(b))


def test_trailing_padding_does_not_change_earlier_logits():
    m = IconSmith(tiny()).eval()
    b = fake_batch(L=12)
    short = m(b["tokens"][:, :7], b["text_emb"], b["exemplars"], b["stroke_bucket"])
    long = m(b["tokens"], b["text_emb"], b["exemplars"], b["stroke_bucket"])
    assert torch.allclose(short, long[:, :6], atol=1e-5)


def test_cached_decoding_matches_full_forward():
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, enc_d_model=64, enc_layers=1,
                      enc_heads=4, n_text_tokens=4, text_scale=19.6)
    m = IconSmith(cfg).eval()
    b = 2
    toks = torch.randint(10, 266, (b, 12))
    toks[:, 0] = 1
    text = torch.randn(b, cfg.text_dim)
    ex = torch.randint(10, 266, (b, cfg.k, 6))
    bucket = torch.zeros(b, dtype=torch.long)
    drop = torch.tensor([False, True])
    full = m(toks, text, ex, bucket, drop_style=torch.zeros(b, dtype=torch.bool), drop_text=drop)
    logits, cache = m.start_decode(toks[:, :1], text, ex, bucket, drop)
    steps = [logits]
    for t in range(1, toks.shape[1] - 1):
        steps.append(m.step_decode(toks[:, t], cache))
    assert torch.allclose(torch.stack(steps, 1), full, atol=1e-4)
