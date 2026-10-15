# AAPL Stock Price Prediction Using LSTM

A time-series forecasting project that uses a Long Short-Term Memory (LSTM) neural network in MATLAB to predict Apple Inc. (AAPL) stock closing prices from historical data.

## Features
- Historical stock price analysis
- LSTM-based time-series forecasting
- Adam optimizer and mean squared error loss
- 80/20 chronological train-test split
- Actual vs. predicted price visualization
- Test-set RMSE evaluation
- Next-step price prediction

## Tech Stack
- MATLAB
- Deep Learning Toolbox
- LSTM Neural Networks
- Time-Series Forecasting

## Model Configuration
| Parameter | Value |
|---|---|
| Input feature | Closing price |
| Sequence length | 20 |
| LSTM hidden units | 100 |
| Training ratio | 80% |
| Testing ratio | 20% |
| Maximum epochs | 100 |
| Batch size | 16 |
| Optimizer | Adam |

## Project Files
- `stock_prediction.m` — MATLAB implementation
- `aapl_us_d.csv` — Historical dataset, if included
- `prediction_results.png` — Actual vs. predicted price plot

## Requirements
- MATLAB
- Deep Learning Toolbox
- A compatible MATLAB release supporting the functions used in the script

## How to Run
1. Clone or download this repository.
2. Open MATLAB and set the repository folder as the current folder.
3. Ensure the CSV file is in the project directory.
4. Run `stock_prediction.m`.

## Results
The script calculates test-set RMSE and generates a plot comparing actual and predicted prices. Add your measured RMSE and results here after running the model.![Actual vs Predicted Stock Prices](prediction_results.png)

## Disclaimer
This project is for educational purposes only. Stock price predictions are uncertain and are not financial advice.
