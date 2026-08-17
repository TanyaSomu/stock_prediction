# Test Cases Documentation — ML Stock Market Forecasting System

This document outlines the test cases implemented in the project's automated verification suite to validate the data preprocessing, model sequence target mapping, error metrics, and numerical robustness of the forecasting pipeline.

The tests are located in `test_suite.py` and can be executed locally.

---

## How to Run the Test Suite

Run the following command from the project root directory:
```bash
python test_suite.py
```

---

## Implemented Test Cases

| Test Case Name | Target File / Function | Description | Input / Mock Data | Expected Assertions / Success Conditions |
| :--- | :--- | :--- | :--- | :--- |
| `test_calculate_indicators_columns` | `2_preprocessing.py`<br>`calculate_indicators` | Verifies that all required technical indicators, lags, and target variables are generated in the returned DataFrame. | A mock DataFrame of 100 days of stock price trend data. | Asserts that expected indicator columns (e.g. `SMA_10`, `RSI_14`, `BB_Upper`, `Lag_1`, `Target`) are present in the resulting columns list. |
| `test_calculate_indicators_values` | `2_preprocessing.py`<br>`calculate_indicators` | Validates pricing values, ensuring that lagged values and tomorrow's forecast targets shift correctly. | A mock DataFrame of 100 days of stock price trend data. | Asserts that `Target` at row $t$ matches `Close` at row $t+1$ and `Lag_1` at row $t$ matches `Close` at row $t-1$. Also verifies that RSI values lie strictly between $0$ and $100$. |
| `test_rsi_division_by_zero_robustness` | `2_preprocessing.py`<br>`calculate_indicators` | Confirms the mathematical robustness of the RSI calculation during a continuous upward price trend (when there are no downward movements). | A mock DataFrame with Close prices strictly increasing (from 100 to 200). | Asserts that no `ZeroDivisionError` is raised, and that RSI values do not contain any infinite (`inf`) or null (`NaN`) elements. |
| `test_lstm_sequence_creation_shapes` | `4_train_lstm.py`<br>`create_sequences` | Validates that sequence construction returns correctly shaped feature matrices ($X$) and target vectors ($y$). | A preprocessed mock dataset with 20 features and a sequence length of 10. | Asserts that the feature matrix shape matches `(samples, seq_length, num_features)` and that the target vector length matches the number of samples. |
| `test_lstm_sequence_target_alignment` | `4_train_lstm.py`<br>`create_sequences` | Verifies that the target price $y$ is correctly aligned to represent day $t+1$ relative to the end of the sequence window (day $t$). | A preprocessed mock dataset with Close prices and a sequence length of 10. | Asserts that `y[0]` (target of first sequence ending at day 9) equals the actual Close price of day 10, guarding against off-by-one forecast window bias. |
| `test_calculate_mape_accuracy` | `3_train_models.py`<br>`calculate_mape` | Validates the precision of Mean Absolute Percentage Error (MAPE) calculations. | A static list of actual prices `[100, 200, 150]` and predicted prices `[110, 190, 150]`. | Asserts that the calculated MAPE equals $5.0\%$ (the average of $10\%$, $5\%$, and $0\%$ absolute errors). |

---

## Sample Test Execution Output

```text
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1786954984.044960   23912 port.cc:153] oneDNN custom operations are on.
......
----------------------------------------------------------------------
Ran 6 tests in 0.038s

OK
```
