import pandas as pd
import numpy as np
import seaborn as sn
from scipy.stats import norm, t
import statsmodels.api as sm
from datetime import datetime

class TradingDataFrame():
    def __init__(self, input_df, time_col, price_cols=[], time_unit=None):
        
        self.time_col = time_col
        self.price_cols = price_cols
        df = input_df.copy()

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
        log_returns = log_data[price_cols].diff()

        # Drop first row
        returns = returns.iloc[1:,:].copy()
        log_returns = log_returns.iloc[1:,:].copy()

        self.data = df
        self.log_data = log_data
        self.returns = returns
        self.log_returns = log_returns
    
    def set_criteria_df(self, df):
        if df.shape != self.data.shape:
            print(f'df shape does not match (data: {self.data.shape}, criteria: {df.shape}), please check.')
        else:
            self.criteria_df = df

    def fit_pdf(self, col='Close_BTC', data='log_returns', distribution='normal'):

        # Choose data
        if data == 'log_returns':
            price_series = self.log_returns[col]
        elif data == 'returns':
            price_series = self.returns[col]
        
        # Choose distribution
        if distribution == 'normal':
            loc, scale = norm.fit(price_series)
        elif distribution == 't':
            s, loc, scale = t.fit(price_series)     

        ax = sn.histplot(price_series, stat='density')
        x0, x1 = ax.get_xlim()  # extract the endpoints for the x-axis
        x_pdf = np.linspace(x0, x1, 100)
        y_pdf = norm.pdf(x_pdf, loc=loc, scale=scale)

        return ax.plot(x_pdf, y_pdf, 'r', lw=2, label='pdf')
    
    def qqplot(self, col='Close_BTC', data='log_returns'):

        # Choose data
        if data == 'log_returns':
            price_series = self.log_returns[col]
        elif data == 'returns':
            price_series = self.returns[col]

        return sm.qqplot(price_series, line='45')