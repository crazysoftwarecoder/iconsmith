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
