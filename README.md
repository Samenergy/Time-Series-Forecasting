
# 🌫️ PM2.5 Time-Series Prediction with LSTM

This project focuses on predicting PM2.5 air pollution levels using time-series deep learning techniques, particularly LSTM-based models. Our best-performing model combines **Bidirectional LSTM**, **Layer Normalization**, **LeakyReLU**, and the **Nadam optimizer**, achieving robust results in forecasting air quality.

## 📊 Problem Statement

Air pollution, specifically PM2.5 particles, poses a significant health risk. Accurate short-term forecasts can support better planning and health responses. This project aims to forecast PM2.5 values from multivariate time-series air quality data using deep learning.

---

## 🧠 Best Model Architecture

```python
model = Sequential([
    Bidirectional(LSTM(64, activation='tanh', return_sequences=True), input_shape=(seq_length, X_train_seq.shape[2])),
    LayerNormalization(),
    Dropout(0.25),

    LSTM(32, activation='tanh'),
    LayerNormalization(),
    Dropout(0.25),

    Dense(64, kernel_regularizer=l2(0.001)),
    LeakyReLU(alpha=0.1),

    Dense(1)
])
```

**Optimizer:** `Nadam`
**Loss Function:** Mean Squared Error
**Final Training Loss (MSE):** `0.01288`
**Final Training RMSE:** `0.05796`

---

## 📁 Project Structure

```
Time-Series-Forecasting/
├── data/                     
├── notebooks/               
│   └── pm25_prediction.ipynb
├── outputs/                  
├── README.md                 
```

---

## 📋 Experiments Summary

| No.   | Architecture                                   | Optimizer | Parameters                           | MSE         | RMSE        |
| ----- | ---------------------------------------------- | --------- | ------------------------------------ | ----------- | ----------- |
| 1     | Vanilla LSTM                                   | Adam      | 2 layers, LR=0.001, Dropout=0.3      | 0.01876     | 0.0682      |
| 2     | Stacked LSTM + Dropout                         | RMSprop   | 3 layers, Dropout=0.3                | 0.01521     | 0.0648      |
| **3** | **Bidirectional LSTM + LayerNorm + LeakyReLU** | **Nadam** | **Dropout=0.25, L2=0.001, LR=0.001** | **0.01288** | **0.05796** |
| 4     | LSTM + Conv1D                                  | Adam      | Conv1D + LSTM combo                  | 0.01389     | 0.05442     |
| 5     | LSTM + Attention                               | Adam      | Vanilla Attention Layer              | 0.01150     | 0.06602     |
| 6     | Conv1D + LSTM Hybrid                           | Adam      | Local+Temporal features              | 0.01538     | 0.04322     |
| 7     | Deep LSTM + DropConnect                        | Adam      | L2=0.001, DropConnect                | 0.00825     | 0.09082     |
| 8     | Lightweight Fast LSTM                          | Adam      | 1-layer optimized for speed          | 0.01288     | 0.05885     |
| 9     | Conv1D + LSTM (Alt Hybrid)                     | Adam      | Modified Conv1D combo                | 0.01271     | 0.05882     |
| 10    | Stacked Bidirectional LSTM                     | Adam      | 3 Bi-LSTM layers                     | 0.01168     | 0.05893     |

---

## 🔧 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Samenergy/Time-Series-Forecasting.git
cd pm2.5-lstm-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the model

```bash
python main.py
```

---

## 📈 Results Visualization

* Time series predictions vs actual PM2.5 values
* Loss and RMSE curves during training
* Architecture comparisons


---

## 🤝 Contribution

Feel free to fork, open issues, or submit PRs if you find ways to improve the model or add new experiments.

---

## 📬 Contact

Built with ❤️ by [Samuel Dushime](https://github.com/Samenergy)

