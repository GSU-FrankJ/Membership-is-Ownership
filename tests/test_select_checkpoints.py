import json
import pathlib

from ddpm_ddim import select_checkpoints


def test_run_selection_ranks_by_auc(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    ckpt_a = run_dir / "ckpt_000100"
    ckpt_b = run_dir / "ckpt_000200"
    ckpt_a.mkdir(parents=True)
    ckpt_b.mkdir(parents=True)
    (ckpt_a / "ema.ckpt").write_bytes(b"ckpt-a")
    (ckpt_b / "ema.ckpt").write_bytes(b"ckpt-b")

    metrics_map = {
        100: {"roc_auc": 0.92, "mean_aux": 1.0, "mean_out": 0.2, "delta_mean": 0.8, "num_aux": 5, "num_out": 5},
        200: {"roc_auc": 0.90, "mean_aux": 1.2, "mean_out": 0.1, "delta_mean": 1.1, "num_aux": 5, "num_out": 5},
    }

    def fake_load_yaml(path: pathlib.Path) -> dict:
        return {
            "model": {},
            "training": {"selection_batch_size": 1},
            "diffusion": {"timesteps": 10},
        }

    def fake_evaluate(ckpt_path, model_cfg, data_cfg, timesteps, sample_limit, device):
        step = int(ckpt_path.parent.name.split("_")[-1])
        metric = metrics_map[step].copy()
        return metric

    monkeypatch.setattr(select_checkpoints, "load_yaml", fake_load_yaml)
    monkeypatch.setattr(select_checkpoints, "evaluate_checkpoint", fake_evaluate)

    model_cfg_path = tmp_path / "model.yaml"
    data_cfg_path = tmp_path / "data.yaml"
    model_cfg_path.write_text("{}", encoding="utf-8")
    data_cfg_path.write_text("{}", encoding="utf-8")

    out_path = select_checkpoints.run_selection(
        run_dir=run_dir,
        model_config=model_cfg_path,
        data_config=data_cfg_path,
        top_k=1,
        timesteps=5,
        sample_limit=10,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["selected"][0]["step"] == 100
    assert len(payload["evaluated"]) == 2
