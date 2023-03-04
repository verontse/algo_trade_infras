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

        # Convert price cols to numbers & add pct_change
        for col in price_cols:
            df[col] = pd.to_numeric(df[col])
            df[col+'_pct_change'] = df[col].pct_change()

        # Set time col to index
        if time_unit != None:
            df[time_col] = pd.to_datetime(df[time_col], unit=time_unit)
        df.set_index(time_col, inplace=True)

        # Drop first row
        df = df.iloc[1:,:]

        self.data = df