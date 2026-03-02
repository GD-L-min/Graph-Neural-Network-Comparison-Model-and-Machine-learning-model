# GNN + ML for Molecular Representation Learning and PCE Prediction

This repository contains two complementary subprojects:

- `GNNCM`: self-supervised contrastive learning on molecular graphs for molecular representation learning, OGB downstream evaluation, similarity-based molecule retrieval, and CAS number lookup.
- `ML`: traditional machine learning regression based on handcrafted molecular descriptors for predicting `Enhanced_PCE`.

If you want a quick overview of this repository, you can think of it as a two-stage research workflow:

1. Use `GNNCM` to learn graph-level molecular representations and validate their quality on public molecular classification benchmarks.
2. Use `ML` to build regression models from structural, electronic, and geometric descriptors for direct property prediction.

## Highlights

- A two-subproject design that covers both graph representation learning and descriptor-based regression.
- A complete self-supervised molecular graph pipeline: `SMILES -> graph -> GCN/GIN contrastive pretraining -> OGB evaluation -> similarity search`.
- Broad coverage of traditional regression models, including linear models, kernel methods, ensemble methods, and boosting models.
- Data and result files are organized by task, making the repository suitable for reproducibility and further extension.
- Released under the MIT License for research reuse.

## Repository Layout

```text
.
├─ GNNCM/
│  ├─ data/
│  │  ├─ train_smiles.txt
│  │  ├─ Anchor Molecules.txt
│  │  └─ 1_million.txt
│  ├─ models/
│  │  ├─ encoders.py
│  │  └─ gcl_model.py
│  ├─ utils/
│  │  ├─ data_preprocessing.py
│  │  └─ augmentations*.py
│  ├─ train_gcn.py / train_gin.py
│  ├─ train_gcn_no_aug.py / train_gin_no_aug.py
│  ├─ evaluate.py / evaluate_aug.py / evaluate_no_aug.py
│  ├─ smiles_similarity_search.py
│  ├─ cas_lookup_from_filtered.py
│  └─ results/
└─ ML/
   ├─ property database.csv
   ├─ Predictive database.csv
   ├─ LinearRegression.py
   ├─ Ridge.py
   ├─ SVR.py
   ├─ RF.py
   ├─ Gaussian Processes.py
   ├─ LightGBM.py
   ├─ XGBoost_Performance.py
   ├─ CatBoost.py
   └─ CatBoost_Predicted.py
```

## Project Snapshot

### `GNNCM`

- Training corpus: `GNNCM/data/train_smiles.txt`, containing `100,000` SMILES strings.
- Search library: `GNNCM/data/1_million.txt`, containing `1,000,001` records.
- Anchor set: `GNNCM/data/Anchor Molecules.txt`, containing `10` molecules.
- Encoders: `GCN`, `GIN`
- Training modes: contrastive learning without augmentation, plus multiple augmentation presets (`0.1 / 0.2 / 0.25 / 0.3`)
- Downstream evaluation: multiple OGB molecular property classification datasets

### `ML`

- Training set: `ML/property database.csv`, containing `190` samples.
- Prediction set: `ML/Predictive database.csv`, containing `128` candidate molecules.
- Input features: 25 descriptor/property columns, including `Initial_PCE`
- Prediction target: `Enhanced_PCE`
- Implemented models: Linear Regression, Ridge, SVR, Random Forest, Gaussian Process, LightGBM, XGBoost, CatBoost

## 1. GNNCM: Graph Contrastive Learning for Molecules

### 1.1 What This Subproject Does

The core goal of `GNNCM` is to learn high-quality molecular graph embeddings. This subproject includes:

- RDKit-based preprocessing from `SMILES` to `PyG` graph objects.
- `GCN` / `GIN` graph contrastive encoders.
- Three graph augmentation strategies: node masking, edge deletion, and subgraph deletion.
- Linear probing evaluation on OGB molecular datasets.
- Large-scale molecular screening based on embedding similarity.
- CAS number completion through PubChem.

### 1.2 Model Pipeline

```text
SMILES
  -> RDKit molecular graph
  -> GCN / GIN encoder
  -> projection head
  -> contrastive objective
  -> graph embedding
  -> downstream evaluation / similarity search
```

### 1.3 Environment Setup

It is recommended to create a separate environment for `GNNCM`. The minimum dependencies are listed in `GNNCM/requirements.txt`:

```bash
cd GNNCM
pip install -r requirements.txt
```

Notes:

- Packages related to `torch-geometric` usually need to match your local `PyTorch + CUDA` version.
- If you want to run `cas_lookup_from_filtered.py`, you also need to install `requests`:

```bash
pip install requests
```

### 1.4 Quick Start

#### Step 1: Preprocess SMILES into graph data

```bash
cd GNNCM
python utils/data_preprocessing.py
```

This script generates:

- `data/processed/processed_data.pkl`
- `data/processed/dataset_stats.pkl`

#### Step 2: Train no-augmentation baselines

```bash
python train_gcn_no_aug.py
python train_gin_no_aug.py
```

#### Step 3: Train contrastive models with augmentation

```bash
python train_gcn.py
python train_gin.py
```

By default, these scripts use the augmentation rates defined in `utils/augmentations_0.3.py`. To switch to `0.1 / 0.2 / 0.25`, modify `CONFIG["augmentation_preset"]` directly.

#### Step 4: OGB downstream evaluation

```bash
python evaluate_no_aug.py
python evaluate_aug.py
```

If you only want to evaluate a specific encoder or augmentation tag, you can use:

```bash
python evaluate_aug.py --encoders gcn --tags augmentations_0_2
python evaluate_aug.py --encoders gin --datasets ogbg-molhiv,ogbg-moltox21
```

#### Step 5: Similarity-based molecule retrieval

Because the actual anchor file in this repository is named `Anchor Molecules.txt`, it is recommended to pass the path explicitly:

```bash
python smiles_similarity_search.py --anchor_smiles "data/Anchor Molecules.txt" --library_smiles "data/1_million.txt" --model_path "path/to/your_checkpoint.pth" --similarity_threshold 0.95
```

#### Step 6: CAS number lookup

```bash
python cas_lookup_from_filtered.py --filtered_txt "results/similarity/Filtered_results_0.95_1_million.txt" --output_csv "results/similarity/Filtered_results_0.95_1_million_cas.csv"
```

## 2. ML: Descriptor-Based Regression for Enhanced PCE

### 2.1 What This Subproject Does

The `ML` subproject uses traditional molecular descriptors to perform supervised regression on `Enhanced_PCE`. Each script is an independent experiment entry point, and the overall workflow is:

```text
CSV descriptors
  -> train/test split
  -> feature standardization
  -> model fitting / grid search
  -> metric report
  -> parity plot with confidence interval
```

### 2.2 Implemented Models

| Script | Model |
| --- | --- |
| `LinearRegression.py` | Linear regression |
| `Ridge.py` | Ridge regression |
| `SVR.py` | Support vector regression |
| `RF.py` | Random forest regression |
| `Gaussian Processes.py` | Gaussian process regression |
| `LightGBM.py` | LightGBM regression |
| `XGBoost_Performance.py` | XGBoost regression |
| `CatBoost.py` | CatBoost regression |
| `CatBoost_Predicted.py` | Post-training inference script |

### 2.3 Environment Setup

The `ML` directory does not include a standalone `requirements.txt`, so install dependencies according to the scripts:

```bash
cd ML
pip install numpy pandas scipy matplotlib statsmodels scikit-learn joblib
pip install lightgbm xgboost catboost
```

### 2.4 Quick Start

#### Train a single model

```bash
cd ML
python CatBoost.py
```

Or run other models:

```bash
python LinearRegression.py
python Ridge.py
python SVR.py
python RF.py
python "Gaussian Processes.py"
python LightGBM.py
python XGBoost_Performance.py
```

#### Predict new samples with CatBoost

```bash
python CatBoost.py
python CatBoost_Predicted.py
```

## 3. License

This project is released under the [MIT License](LICENSE.txt).
