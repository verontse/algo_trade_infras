import pandas as pd
import numpy as np
from datetime import datetime

class TradingDataFrame():
    def __init__(self, df, time_col, price_criteria_pair=None, price_cols=[], time_unit=None):
        
        self.time_col = time_col
        self.price_criteria_pair = price_criteria_pair
        self.price_cols = price_cols

        if len(price_cols) == 0:
            self.price_cols = [price for price in price_criteria_pair.keys()]

        # Set time col to index
        if time_unit != None:
            df[time_col] = pd.to_datetime(df[time_col], unit=time_unit)
        df.set_index(time_col, inplace=True)

        # Convert price cols to numbers
        for col in price_cols:
            df[col] = pd.to_numeric(df[col])
        
        # make log returns, pct_change and log_pct_change df
        returns = df[price_cols].pct_change()

        df_plusone = df.copy() + 1
        log_data = np.log(df_plusone)
        log_returns = log_data[price_cols].pct_change()

        # Drop first row
        returns = returns.iloc[1:,:].copy()
        log_returns = log_returns.iloc[1:,:].copy()

        self.data = df
        self.log_data = log_data
        self.returns = returns
        self.log_returns = log_returns