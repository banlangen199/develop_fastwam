from __future__ import annotations

import argparse
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from fastwam.utils.config_resolvers import register_default_resolvers


def _config_name(path: str) -> str:
    return Path(path).stem


def _write_shapes(path: Path, report: dict[str, dict], *, is_model: bool):
    cfg = OmegaConf.load(path)
    with open_dict(cfg):
        if is_model:
            if "dream_query_config" not in cfg or cfg.dream_query_config is None:
                cfg.dream_query_config = {}
            if "dream_decoder" not in cfg.dream_query_config or cfg.dream_query_config.dream_decoder is None:
                cfg.dream_query_config.dream_decoder = {}
            decoder = cfg.dream_query_config.dream_decoder
            for modality, item in report.items():
                if modality not in decoder or decoder[modality] is None:
                    decoder[modality] = {}
                decoder[modality]["target_layout"] = str(item["target_layout"])
                decoder[modality]["target_shape"] = list(item["final_decoder_target_shape"])
        else:
            if "train" not in cfg or cfg.train is None:
                cfg.train = {}
            if "dream_target" not in cfg.train or cfg.train.dream_target is None:
                cfg.train.dream_target = {}
            target = cfg.train.dream_target
            if "target_shapes" not in target or target.target_shapes is None:
                target.target_shapes = {}
            for modality, item in report.items():
                target["target_shapes"][modality] = list(item["final_decoder_target_shape"])
                if modality not in target or target[modality] is None:
                    target[modality] = {}
                target[modality]["target_layout"] = str(item["target_layout"])
    OmegaConf.save(cfg, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True, help="Path like configs/data/libero_spatial_2cam.yaml")
    parser.add_argument("--model-config", required=True, help="Path like configs/model/dream_fastwam.yaml")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--write-config", action="store_true")
    args = parser.parse_args()

    register_default_resolvers()
    repo_root = Path(__file__).resolve().parents[1]
    config_root = repo_root / "configs"
    data_name = _config_name(args.data_config)
    model_name = _config_name(args.model_config)

    with hydra.initialize_config_dir(config_dir=str(config_root), version_base="1.3"):
        cfg = hydra.compose(
            config_name="train",
            overrides=[f"data={data_name}", f"model={model_name}"],
        )
    dataset = instantiate(cfg.data.train)
    report = dataset.discover_dream_target_shapes(args.idx)
    print(report)

    if args.write_config:
        data_path = Path(args.data_config)
        model_path = Path(args.model_config)
        _write_shapes(data_path, report, is_model=False)
        _write_shapes(model_path, report, is_model=True)
        print(f"Wrote target shapes to {data_path} and {model_path}")


if __name__ == "__main__":
    main()
