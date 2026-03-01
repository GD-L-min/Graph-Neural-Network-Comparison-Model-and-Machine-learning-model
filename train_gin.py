
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
import pickle
import os
import json
from tqdm import tqdm
from pathlib import Path
import importlib.util

from models.gcl_model import GCLModel
from utils.augmentations import Augmentation


def load_augmentation_preset(preset_name: str):
    """从 utils/augmentations_*.py 动态加载增强比例。

    preset_name 示例："augmentations_0.1"（对应 utils/augmentations_0.1.py）
    返回：dict[str, float]
    """
    if not preset_name:
        return None

    utils_dir = Path(__file__).resolve().parent / "utils"
    preset_path = utils_dir / f"{preset_name}.py"
    if not preset_path.exists():
        raise FileNotFoundError(f"找不到增强预设文件：{preset_path}")

    module_name = preset_name.replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, str(preset_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载增强预设：{preset_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "get_rates"):
        return module.get_rates()
    if hasattr(module, "AUGMENTATION_RATES"):
        return dict(module.AUGMENTATION_RATES)

    raise AttributeError(f"增强预设文件中未找到 get_rates 或 AUGMENTATION_RATES：{preset_path}")


# --- 配置 --- #
CONFIG = {
    "encoder_name": "gin",
    "input_dim": 19,
    "hidden_dim": 256,
    "projection_dim": 128,
    "num_layers": 5,
    "dropout": 0.5,
    "temperature": 0.1,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 512,
    "epochs": 100,

    "augmentation_methods": ["node_masking", "edge_deletion", "subgraph_deletion"],

    # 新增：可自主选用增强比例（四个预设）
    # 设为空字符串/None：使用 utils/augmentations.py 中的默认比例 0.15
    "augmentation_preset": "augmentations_0.3",
    "augmentation_presets": ["augmentations_0.1", "augmentations_0.2", "augmentations_0.25", "augmentations_0.3"],

    # 兼容：如果 augmentation_preset 为空，则回退用这里的自定义比例
    # 设为空 dict：让 Augmentation 对每种方法使用默认 0.15
    "augmentation_rates": {},

    "data_path": r"C:\Users\LGD\Desktop\cursor\GNN-12.10\molecular_gnn_final\data\processed\processed_data.pkl",
    "save_dir": r"C:\Users\LGD\Desktop\cursor\GNN-12.10\molecular_gnn_final\results\checkpoints",
    "log_dir": r"C:\Users\LGD\Desktop\cursor\GNN-12.10\molecular_gnn_final\logs\tensorboard"
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.benchmark = True

    # --- Build augmentation tag for saving artifacts ---
    preset_name = CONFIG.get("augmentation_preset")
    if preset_name:
        aug_tag = preset_name.replace(".", "_")
    else:
        # augmentation_rates 为空 dict 时，会使用 Augmentation 的默认值（0.15）
        aug_tag = "aug_default_0_15" if not CONFIG.get("augmentation_rates") else "aug_custom"

    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    with open(os.path.join(CONFIG["save_dir"], f"config_gin_{aug_tag}.json"), 'w') as f:
        json.dump(CONFIG, f, indent=4)

    print("Loading preprocessed data...")
    with open(CONFIG["data_path"], 'rb') as f:
        train_data = pickle.load(f)

    train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True)
    print(f"Data loaded. Number of graphs: {len(train_data)}, Number of batches: {len(train_loader)}")

    print(f"Initializing {CONFIG['encoder_name'].upper()} based GCL model...")
    model = GCLModel(
        encoder_name=CONFIG["encoder_name"],
        input_dim=CONFIG["input_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        projection_dim=CONFIG["projection_dim"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        temperature=CONFIG["temperature"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

    preset_rates = load_augmentation_preset(CONFIG.get("augmentation_preset"))
    augmentation_rates = preset_rates if preset_rates is not None else CONFIG["augmentation_rates"]
    augmentation = Augmentation(CONFIG["augmentation_methods"], augmentation_rates)

    writer = SummaryWriter(log_dir=os.path.join(CONFIG["log_dir"], CONFIG["encoder_name"]))
    loss_history = []

    print("Starting training...")
    min_loss = float('inf')

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
        for batch_data in progress_bar:
            optimizer.zero_grad()

            data1, data2 = augmentation(batch_data)
            data1, data2 = data1.to(device), data2.to(device)

            p1, p2 = model(data1, data2)
            loss = model.contrastive_loss(p1, p2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        loss_path = os.path.join(CONFIG["save_dir"], f"{CONFIG['encoder_name']}_{aug_tag}_loss_history.json")
        with open(loss_path, 'w') as lf:
            json.dump(loss_history, lf, indent=4)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f"Epoch {epoch}/{CONFIG['epochs']}, Average Loss: {avg_loss:.4f}")

        if avg_loss < min_loss:
            min_loss = avg_loss
            save_path = os.path.join(CONFIG["save_dir"], f"{CONFIG['encoder_name']}_{aug_tag}_best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path} with loss {avg_loss:.4f}")

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()

