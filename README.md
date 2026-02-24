# Bitcoin Closing Price Prediction

A PyTorch-based neural network that predicts the next-day closing price of Bitcoin using historical daily market data from 2014 to 2023.

## Overview

This project builds a fully connected feedforward neural network to forecast Bitcoin's next-day closing price. The model is trained on 9 years of daily price data (2014–2023) sourced from [Kaggle](https://www.kaggle.com/datasets/omarshahrukh/bitcoin-daily-prices-2014-2023-9-years).

## Dataset

The dataset (`data/Bitcoin_Price_Dataset_2014_2023.csv`) contains **3 393 daily records** with 17 features:

| Feature | Description |
|---|---|
| `Date` | Trading date |
| `Open`, `High`, `Low`, `Close` | Daily OHLC prices (USD) |
| `Volume` | Trading volume |
| `Daily_Return` | Percentage daily return |
| `Price_Range` | High − Low |
| `Price_Change` | Close − Open |
| `MA_7`, `MA_30`, `MA_90` | Moving averages (7 / 30 / 90 days) |
| `Volatility_30d` | 30-day rolling volatility |
| `Day_of_Week` | Day name (dropped before training) |
| `Month`, `Year`, `Quarter` | Calendar features |

The **target variable** is the next day's closing price (`Close` shifted by −1).

## Model Architecture

```
Input (15 features)
  → Linear(15, 64) → ReLU
  → Linear(64, 32) → ReLU
  → Linear(32, 1)
Output (predicted closing price)
```

## Training Details

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.01 |
| Loss function | MSE (Mean Squared Error) |
| Batch size | 32 |
| Epochs | 100 |
| Train / Val / Test split | 80% / 10% / 10% (chronological, no shuffle) |
| Feature scaling | `StandardScaler` (fit on train set only) |

## Evaluation Metrics

Reported on the **test set** (scaled and real-world):

- **MSE** (scaled)
- **MAE** (scaled & USD)
- **R² Score**

## Visualizations

The notebook produces four plots:

1. **Bitcoin Close Price (2014–2023)** — full historical price chart.
2. **Dataset Split Sizes** — treemap showing train / validation / test sizes.
3. **Learning Curve** — training vs. validation loss over epochs.
4. **Actual vs. Predicted** — true and predicted prices on the test set.

## Project Structure

```
bitcoin-prices/
├── bitcoin_prices.ipynb          # Main notebook (data loading → training → evaluation → plots)
├── requirements.txt              # Python dependencies
├── README.md
└── data/
    └── Bitcoin_Price_Dataset_2014_2023.csv
```

## Getting Started

### Prerequisites

- Python 3.10+
- (Optional) CUDA-compatible GPU for faster training

### Installation

```bash
git clone <repo-url>
cd bitcoin-prices
pip install -r requirements.txt
```

### Running

Open `bitcoin_prices.ipynb` in Jupyter / PyCharm / VS Code and run all cells.  
The first cell downloads the dataset via `kagglehub` (requires a Kaggle account and API token).

## License

This project is provided for educational purposes.

