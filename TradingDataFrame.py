import pandas as pd
import numpy as np
from datetime import datetime

class TradingDataFrame():
    def __init__(self, df, time_col, price_cols, criteria_cols, source):
        
        self.time_col = time_col
        self.price_cols = price_cols
        self.criteria_cols = criteria_cols
        
        cols = [time_col, *price_cols, *criteria_cols]
        df = df[cols].copy()
        
        # Convert price cols to numbers & add pct_change
        for col in price_cols:
            df[col] = pd.to_numeric(df[col])
            df[col+'_pct_change'] = df[col].pct_change()

        # Set time col to index
        if source == 'binance':
            df[time_col] = pd.to_datetime(df[time_col], unit='ms')
        df.set_index(time_col, inplace=True)

        # Drop first row
        df = df.iloc[1:,:]

        self.data = df

print('hello')