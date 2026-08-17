![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Architecture](https://img.shields.io/badge/Model-Bi--LSTM%20%2B%20Transformer-orange.svg)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9742-brightgreen.svg)

# 🌫️ Technical Report & Operational Guide: Hybrid LSTM-Transformer Architecture for Operational Winter Fog Prediction

**Author**: Shaurya P.S. Yadav (251340011)  
**System Package**: `fog_prediction_system`  
**Target Location**: Lucknow Region (Station ID: 42369099999)  

---

## Executive Summary

Winter fog is a severe meteorological hazard across Northern India, causing major disruptions to transportation, aviation safety, and power grids. Dense radiation fog develops rapidly during winter months (November through February), reducing surface visibility below critical operational safety thresholds.

This repository encapsulates `fog_prediction_system`, a modular Python framework engineered for **6-hour** and **12-hour ahead operational fog forecasting**. The framework integrates high-resolution ERA5 meteorological reanalysis with ground-truth aviation visibility observations (2000–2025), utilizing a novel **Hybrid LSTM Encoder + Multi-Head Transformer** deep learning architecture.

---

## 1. Problem Formulation & Atmospheric Criteria

Winter fog prediction is formulated as a supervised binary classification task:
$$\text{Fog} = \begin{cases} 1 & \text{if } \Delta T \le 2\text{ K AND } V \le 2\text{ m/s} \\ 0 & \text{otherwise} \end{cases}$$

Where:
- $\Delta T = T - T_d$ is **Dew Point Depression**, representing near-surface atmospheric moisture saturation ($\le 2\text{ K}$).
- $V$ is **Surface Wind Speed**, representing weak boundary layer turbulent mixing ($\le 2\text{ m/s}$) required for fog accumulation and persistence.

### Key Modeling Challenges
1. **Nonlinear Atmospheric Complexity**: Fog formation depends on delicate balances between radiative cooling, temperature inversions, relative humidity, and surface pressure.
2. **Severe Class Imbalance**: Historical observations reveal that fog events constitute only **~3.6%** of total observations.

---

## 2. System Architecture & Modular Design

The codebase is organized into a clean, modular Python package (`fog_prediction_system`):

```
fog_prediction_system/
├── config.py                 # Centralized configuration (paths, hyperparameters, thresholds)
├── main.py                   # Unified CLI entry point (--mode full|build_data|eda|train)
├── requirements.txt          # Dependencies (numpy, pandas, scikit-learn, torch, matplotlib)
├── README.md                 # System technical report & operational guide
├── data/
│   ├── dataset_builder.py    # Merges ERA5 reanalysis and NOAA ISD yearly records
│   └── loader.py             # Time-series loading and schema validation
├── processing/
│   ├── eda_missing.py        # Missing value filter (>70%) & past-only 3-day median rolling imputation
│   ├── feature_engineering.py# Cyclic (sin/cos), lag, rolling window, & physical interaction features
│   └── sequence.py           # 3D sliding window tensor builder & temporal data splitter
├── models/
│   ├── layers.py             # LayerNorm, LSTMLayer, MultiHeadAttention, & TransformerBlock
│   └── lstm_transformer.py   # Hybrid LSTM + Transformer model (NumPy & PyTorch backends)
├── training/
│   ├── trainer.py            # Adam optimizer, mini-batch training loop, & val checkpointing
│   └── evaluator.py          # NSE, KGE, R², ROC-AUC, PR-AUC, F1, Precision, & Recall
└── visualization/
    └── plots.py              # Visualizations for missing values, cyclic features, heatmaps, & curves
```

---

## 3. Feature Engineering & Visualizations

Raw predictors are transformed into a 24-dimensional feature representation capturing atmospheric memory, diurnal cycles, and physical interactions:

1. **Cyclic Markers**: Hours, months, and days-of-year mapped onto unit circle coordinates using $\sin/\cos$ transformations to ensure continuity across midnight and seasonal transitions.
2. **Atmospheric Persistence**: Multi-step temporal lags ($2\text{h}, 6\text{h}, 12\text{h}, 24\text{h}$) capturing thermodynamic trends.
3. **Domain Interaction Terms**: Humidity $\times$ Temperature, Relative Humidity $\times$ Wind Speed, Dew Point Depression $\times$ Humidity, Ceiling / Visibility ratios.
4. **Rolling Window Statistics**: 6-hour, 12-hour, and 24-hour rolling means, standard deviations, and maximums.

### Exploratory & Operational Visualizations

#### 1. Feature Distributions: Fog vs. No-Fog
![Feature Distributions](assets/Feature%20distribution.png)
*Figure 1: Comparison of key atmospheric feature distributions during Fog (red) and No-Fog (blue) events.*

> **Description & Inferences**:  
> The plot illustrates clear physical separation between classes across key predictors. Relative humidity (`rh_pct`) during fog events is heavily concentrated above $90\%$, whereas No-Fog events exhibit a broad distribution. Dew point depression (`td_depression`) clusters tightly near $0^\circ\text{C}$, confirming atmospheric moisture saturation as an absolute prerequisite. Furthermore, wind speed (`wind_speed`) peaks below $2\text{ m/s}$, validating the weak turbulent mixing requirement necessary for fog accumulation and boundary-layer persistence.

#### 2. Cyclic Sine/Cosine Feature Encodings
![Cyclic Feature Encoding](assets/cyclic_encoding.png)
*Figure 2: Trigonometric unit-circle mapping ($\sin/\cos$) for hour-of-day, month-of-year, and day-of-year.*

> **Description & Inferences**:  
> Standard integer representations of time introduce artificial numerical discontinuities between consecutive time steps (e.g., 23:00 to 00:00 or December to January). By mapping temporal variables onto unit circles using sine and cosine transformations, cyclical continuity is preserved. This encoding allows the Transformer multi-head attention mechanism to seamlessly capture diurnal radiative cooling cycles and seasonal patterns without boundary artifacts.

#### 3. Temporal Fog Predictions vs. Observed Actuals
![Prediction Dynamics](assets/Prediction.png)
*Figure 3: Model predicted probability dynamics (blue curve) against ground-truth fog observations (black markers) over the test period.*

> **Description & Inferences**:  
> The operational timeline compares the model's predicted fog probability against actual ground-truth fog occurrences over multiple winter seasons. The model maintains a stable near-zero baseline during clear periods and rapidly spikes above the operational safety threshold ($0.8$) prior to fog onset. The high correlation ($R^2 = 0.8608$) confirms that the hybrid architecture effectively captures atmospheric persistence and triggers reliable early warnings.

---

## 4. Deep Learning Model Architecture

The core model is a **Hybrid Bi-LSTM + Multi-Head Transformer Encoder** (181,569 trainable parameters):

```
Input Tensor (Batch, 12 timesteps, 24 features)
       │
       ▼
┌───────────────────────────────────────────────┐
│ Input Projection (Dense 24 → 64, LayerNorm)   │
└───────────────────────┬───────────────────────┘
                        │
┌───────────────────────▼───────────────────────┐
│ Bidirectional LSTM Encoder (Hidden Dim = 64)   │
│ Captures sequential persistence & dynamics     │
└───────────────────────┬───────────────────────┘
                        │
┌───────────────────────▼───────────────────────┐
│ Multi-Head Self-Attention Transformer Block   │
│ (4 Attention Heads, d_ff = 128)               │
│ Attends across all past timesteps             │
└───────────────────────┬───────────────────────┘
                        │
┌───────────────────────▼───────────────────────┐
│ Global Mean Pooling (Batch, 64)               │
└───────────────────────┬───────────────────────┘
                        │
┌───────────────────────▼───────────────────────┐
│ MLP Classification Head (Dense 64→32 → 32→1)  │
│ Output Probability P(Fog) ∈ [0, 1]             │
└───────────────────────────────────────────────┘
```

---

## 5. Performance Metrics & Evaluation

The system is evaluated using both **Classification Metrics** and **Hydrological Model Efficiency Metrics**:

| Metric | Target (6-Hour Ahead) | Target (12-Hour Ahead) | Description |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | **0.9742** | **0.9581** | Area under Receiver Operating Characteristic curve |
| **PR-AUC** | **0.8915** | **0.8432** | Precision-Recall AUC (critical for imbalanced classes) |
| **NSE (Nash-Sutcliffe Efficiency)** | **0.8608** | **0.8145** | Hydrological variance explained ($1.0 = \text{perfect}$) |
| **KGE (Kling-Gupta Efficiency)** | **0.8824** | **0.8310** | Multi-objective correlation, bias, and variability score |
| **$R^2$ Score** | **0.8608** | **0.8145** | Variance explained ($86\%$ variance captured at 6h) |

---

## 6. Installation & How to Use

### Prerequisites & Dependencies
Install package dependencies:
```bash
pip install -r fog_prediction_system/requirements.txt
```

### CLI Command Options

#### 1. Full Pipeline Execution
Runs data loading, imputation, feature engineering, sequence tensor creation, training (6h & 12h models), and evaluation:
```bash
python -m fog_prediction_system.main --mode full
```

#### 2. Raw Dataset Compilation
Compiles raw ERA5 reanalysis and yearly NOAA ISD files into `visuals/fog_dataset_2000_2025.csv`:
```bash
python -m fog_prediction_system.main --mode build_data
```

#### 3. EDA & Imputation Mode
Processes missing values (>70% drop threshold, 3-day past rolling median fill):
```bash
python -m fog_prediction_system.main --mode eda
```

#### 4. Python Programmatic Usage
You can also import and use modules programmatically in your own scripts:
```python
from fog_prediction_system.data import load_dataset
from fog_prediction_system.processing import FeatureEngineer, process_missing_values
from fog_prediction_system.models import NumPyLSTMTransformer

# 1. Load Data
df = load_dataset("visuals/fog_dataset_2000_2025.csv")

# 2. Process Missing Values
df_clean = process_missing_values(df)

# 3. Engineer Features
engineer = FeatureEngineer(top_k=25)
df_features, selected_cols = engineer.process(df_clean)

# 4. Instantiate Model
model = NumPyLSTMTransformer(in_dim=len(selected_cols))
print("Model initialized successfully!")
```

---

## 7. Conclusion

The modular `fog_prediction_system` provides a robust, reproducible, and operational solution for localized winter fog forecasting over the Lucknow region. By integrating sequential memory with multi-head self-attention mechanisms, the system successfully addresses non-linear thermodynamic interactions and severe class imbalance.
