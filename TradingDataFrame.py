import pandas as pd
import numpy as np
from datetime import datetime

class TradingDataFrame():
    def __init__(self, df, time_col, price_criteria_pair, time_unit=None):
        
        self.time_col = time_col
        self.price_criteria_pair = price_criteria_pair
        
        # Remove unnecessary columns
        price_cols, criteria_cols = [], []
        for price_col, criteria_col in price_criteria_pair.items():
            price_cols.append(price_col)
            criteria_cols.append(criteria_col)

        criteria_cols = list(set(criteria_cols))
        
        cols = [time_col, *price_cols, *criteria_cols]
        df = df[cols].copy()

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