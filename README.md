# Stock Price Prediction Using RNNs

This project predicts Amazon stock closing prices using historical stock-market data from four technology companies: Amazon, Google, IBM, and Microsoft. The notebook builds a recurrent neural network pipeline for time-series forecasting and compares a Simple RNN model with an LSTM model.

## Project Objective

Given historical daily stock data for AMZN, GOOGL, IBM, and MSFT, the goal is to predict the next closing price for Amazon using a fixed-size window of previous trading days.

## Dataset

The project uses four CSV files, one for each company:

- `AMZN_stocks_data.csv`
- `GOOGL_stocks_data.csv`
- `IBM_stocks_data.csv`
- `MSFT_stocks_data.csv`

Each dataset contains daily stock records from January 1, 2006 to January 1, 2018 with the following columns:

- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`
- `Name`

The notebook aggregates the four datasets into a single master dataframe by suffixing each feature with its company ticker, such as `CloseAMZN`, `VolumeGOOGL`, and `OpenMSFT`.

## Workflow

1. Load and aggregate stock data from all four companies.
2. Handle missing values by removing rows with extensive missing data and filling remaining missing values with column medians.
3. Explore stock volume distributions, volume trends over time, and feature correlations.
4. Convert the time-series data into rolling windows for supervised learning.
5. Scale input windows using `StandardScaler` and target values using `MinMaxScaler`.
6. Train and tune a Simple RNN model.
7. Train and tune an LSTM model.
8. Compare model performance using RMSE and MAE.

## Models

### Simple RNN

The Simple RNN model uses:

- `SimpleRNN` layer with tunable units, learning rate, dropout, and window size
- `Dense` output layer for single-value regression
- Adam optimizer
- Mean squared error loss
- Mean absolute error metric
- Early stopping and learning-rate reduction callbacks

Best tuned configuration:

- Units: `50`
- Learning rate: `0.0005`
- Window size: `45`
- Dropout: `0.0`

Final Simple RNN performance:

- RMSE: `135.11`
- MAE: `92.08`

### LSTM

The LSTM model uses:

- `LSTM` layer with tunable units, learning rate, dropout, and window size
- `Dense` output layer for single-value regression
- Adam optimizer
- Mean squared error loss
- Mean absolute error metric
- Early stopping callback

Best tuned configuration:

- Units: `100`
- Learning rate: `0.001`
- Window size: `30`
- Dropout: `0.0`

Final LSTM performance:

- RMSE: `89.87`
- MAE: `59.96`

The LSTM model outperformed the Simple RNN model, producing lower prediction error on the test dataset.

## Key Insights

- Stock-price features across technology companies showed strong correlations.
- Volume features were less correlated than price-based features.
- Windowed sequence modeling allowed the models to learn from recent historical stock behavior.
- LSTM performed better than Simple RNN, likely because it handles longer-term temporal dependencies more effectively.

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Keras

## How to Run

1. Install the required Python libraries:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
   ```

2. Place the stock CSV files inside a folder named `RNN_Stocks_Data`.

3. Open the notebook:

   ```bash
   jupyter notebook RNN_Stock_Price_Prediction.ipynb
   ```

4. Run the notebook cells in order.

## Repository Structure

```text
.
├── RNN_Stock_Price_Prediction.ipynb
├── README.md
└── RNN_Stocks_Data/
    ├── AMZN_stocks_data.csv
    ├── GOOGL_stocks_data.csv
    ├── IBM_stocks_data.csv
    └── MSFT_stocks_data.csv
```

## Results

| Model | Window Size | Units | Learning Rate | RMSE | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Simple RNN | 45 | 50 | 0.0005 | 135.11 | 92.08 |
| LSTM | 30 | 100 | 0.001 | 89.87 | 59.96 |

The final LSTM model achieved the best results among the tested recurrent neural network architectures.
