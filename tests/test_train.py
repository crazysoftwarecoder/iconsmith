import csv

from iconsmith.model import ModelConfig
from iconsmith.train import TrainConfig, train


def test_train_smoke_writes_artifacts(processed, tmp_path):
    cfg = TrainConfig(
        processed=processed, run_dir=tmp_path / "run", epochs=1, batch_size=4,
        max_steps=30, overfit_n=8, bf16=False, device="cpu", log_every=10, eval_every=15,
    )
    mcfg = ModelConfig(
        d_model=64, n_layers=2, n_heads=4, d_ff=128,
        enc_d_model=32, enc_layers=1, enc_heads=4, max_len=256,
    )
    res = train(cfg, mcfg)
    assert res["steps"] == 30 and res["tokens_per_sec"] > 0
    rows = list(csv.DictReader(open(tmp_path / "run" / "log.csv")))
    assert len(rows) >= 3
    assert (tmp_path / "run" / "ckpt_last.pt").exists()
    assert (tmp_path / "run" / "loss.png").exists()
    assert float(rows[-1]["train_loss"]) < float(rows[0]["train_loss"])
    assert any(r["val_loss"] not in ("", None) for r in rows)
