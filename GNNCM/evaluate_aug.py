import argparse
import json
import os
import re
from dataclasses import dataclass

import torch
from torch_geometric.loader import DataLoader

import evaluate as base_eval
from models.gcl_model import GCLModel


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_DIR = os.path.join(ROOT_DIR, "results", "checkpoints")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "results", "aug_evaluations")


@dataclass(frozen=True)
class RunArtifact:
    encoder_name: str
    aug_tag: str
    run_name: str
    config_path: str
    checkpoint_path: str


def _safe_read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_runs(checkpoint_dir: str) -> list[RunArtifact]:
    """扫描 results/checkpoints，自动发现带 aug_tag 的模型与配置。"""
    if not os.path.isdir(checkpoint_dir):
        print(f"Checkpoint 目录不存在：{checkpoint_dir}")
        return []

    # 新命名：gcn_<aug_tag>_best_model.pth / gin_<aug_tag>_best_model.pth
    ckpt_pattern = re.compile(r"^(gcn|gin)_(.+)_best_model\.pth$")

    runs: list[RunArtifact] = []
    for filename in os.listdir(checkpoint_dir):
        match = ckpt_pattern.match(filename)
        if not match:
            continue

        encoder = match.group(1)
        aug_tag = match.group(2)
        checkpoint_path = os.path.join(checkpoint_dir, filename)

        # 训练脚本保存的配置名：config_gcn_<aug_tag>.json / config_gin_<aug_tag>.json
        config_filename = f"config_{encoder}_{aug_tag}.json"
        config_path = os.path.join(checkpoint_dir, config_filename)

        # 兼容旧文件名（如果用户手动改过/历史遗留）
        if not os.path.isfile(config_path):
            legacy_candidates = [
                os.path.join(checkpoint_dir, "config_gcn.json") if encoder == "gcn" else os.path.join(checkpoint_dir, "config.json"),
                os.path.join(checkpoint_dir, f"config_{encoder}.json"),
            ]
            config_path = next((p for p in legacy_candidates if os.path.isfile(p)), "")

        if not config_path:
            print(f"跳过 {filename}：未找到对应 config（期望 {config_filename}）")
            continue

        run_name = f"{encoder}_{aug_tag}"
        runs.append(
            RunArtifact(
                encoder_name=encoder,
                aug_tag=aug_tag,
                run_name=run_name,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
            )
        )

    runs.sort(key=lambda r: (r.encoder_name, r.aug_tag))
    return runs


def evaluate_run(run: RunArtifact, device: torch.device, output_dir: str, dataset_names: list[str] | None):
    config = _safe_read_json(run.config_path)

    print("\n" + "=" * 80)
    print(f"Evaluating augmentation run: {run.run_name}")
    print(f"Config: {run.config_path}")
    print(f"Checkpoint: {run.checkpoint_path}")
    print("=" * 80)

    model = GCLModel(
        encoder_name=config["encoder_name"],
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        projection_dim=config["projection_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        temperature=config["temperature"],
    ).to(device)
    model.load_state_dict(torch.load(run.checkpoint_path, map_location=device))
    model.eval()

    run_output_dir = os.path.join(output_dir, run.run_name)
    os.makedirs(run_output_dir, exist_ok=True)

    dataset_results: dict[str, dict] = {}
    ogb_datasets = base_eval.load_ogb_datasets(dataset_names=dataset_names)

    for dataset_name, ogb_dataset, split_idx, dataset_dir in ogb_datasets:
        print("\n" + "-" * 80)
        print(f"Dataset: {dataset_name}")
        print("-" * 80)

        try:
            smiles_map = base_eval.load_smiles_mapping(dataset_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {dataset_name}: {exc}")
            continue

        rdkit_splits, skip_stats = base_eval.build_rdkit_feature_aligned_splits(
            ogb_dataset, split_idx, smiles_map
        )

        for split_name in ["train", "valid", "test"]:
            split_dataset = rdkit_splits.get(split_name)
            kept = len(split_dataset) if split_dataset is not None else 0
            skipped = skip_stats.get(split_name, 0)
            print(f"{split_name.capitalize()} split -> kept {kept} graphs, skipped {skipped} graphs")

        missing = [name for name in ["train", "valid", "test"] if len(rdkit_splits.get(name, [])) == 0]
        if missing:
            print(f"{dataset_name} has no RDKit graphs for {', '.join(missing)}; skipping dataset.")
            continue

        train_loader = DataLoader(rdkit_splits["train"], batch_size=256, shuffle=False)
        valid_loader = DataLoader(rdkit_splits["valid"], batch_size=256, shuffle=False)
        test_loader = DataLoader(rdkit_splits["test"], batch_size=256, shuffle=False)

        print("\nExtracting embeddings from train set...")
        train_embeddings, train_labels = base_eval.extract_embeddings(model, train_loader, device)
        print("Extracting embeddings from valid set...")
        valid_embeddings, valid_labels = base_eval.extract_embeddings(model, valid_loader, device)
        print("Extracting embeddings from test set...")
        test_embeddings, test_labels = base_eval.extract_embeddings(model, test_loader, device)

        print("\n" + "=" * 80)
        print("Evaluating classification performance...")
        print("=" * 80)

        print("\nValidation set results:")
        valid_results = base_eval.evaluate_classification(
            train_embeddings, train_labels, valid_embeddings, valid_labels
        )
        for key, value in valid_results.items():
            if isinstance(value, dict):
                continue
            print(f"{key}: {value:.4f}")

        print("\nTest set results:")
        test_results = base_eval.evaluate_classification(
            train_embeddings, train_labels, test_embeddings, test_labels
        )
        for key, value in test_results.items():
            if isinstance(value, dict):
                continue
            print(f"{key}: {value:.4f}")

        dataset_results[dataset_name] = {"valid": valid_results, "test": test_results}

        embeddings_path = os.path.join(run_output_dir, f"{dataset_name}_embeddings.npz")
        base_eval.np.savez(
            embeddings_path,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            valid_embeddings=valid_embeddings,
            valid_labels=valid_labels,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
        )
        print(f"Embeddings saved to {embeddings_path}")

    results_payload = {
        "run": {
            "run_name": run.run_name,
            "encoder_name": run.encoder_name,
            "aug_tag": run.aug_tag,
            "config_path": run.config_path,
            "checkpoint_path": run.checkpoint_path,
        },
        "datasets": dataset_results,
    }

    results_path = os.path.join(run_output_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as results_file:
        json.dump(results_payload, results_file, indent=4)
    print(f"\nAggregated results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate models trained with different augmentation rates.")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR, help="模型与配置所在目录")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="评估输出目录")
    parser.add_argument(
        "--encoders",
        default="gcn,gin",
        help="评估哪些编码器（逗号分隔），例如 gcn 或 gin 或 gcn,gin",
    )
    parser.add_argument(
        "--tags",
        default="",
        help="只评估指定 aug_tag（逗号分隔）。留空表示评估自动发现的全部。",
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="只评估指定 OGB 数据集名（逗号分隔），例如 ogbg-molhiv,ogbg-moltox21；留空表示全部默认集合。",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)

    allow_encoders = {e.strip() for e in args.encoders.split(",") if e.strip()}
    # 兼容：允许用户输入 augmentations_0.25 或 augmentations_0_25
    allow_tags = {t.strip().replace(".", "_") for t in args.tags.split(",") if t.strip()}
    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()] or None

    runs = discover_runs(args.checkpoint_dir)
    if not runs:
        print("未发现任何可评估的模型。")
        print("提示：确认训练已生成形如 gcn_<aug_tag>_best_model.pth 的文件。")
        return

    selected = []
    for run in runs:
        if allow_encoders and run.encoder_name not in allow_encoders:
            continue
        if allow_tags and run.aug_tag not in allow_tags:
            continue
        selected.append(run)

    if not selected:
        print("根据过滤条件未匹配到任何模型。")
        return

    for run in selected:
        evaluate_run(run, device=device, output_dir=args.output_dir, dataset_names=dataset_names)


if __name__ == "__main__":
    main()
