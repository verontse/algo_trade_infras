import pandas as pd
import numpy as np
from datetime import datetime
from TradingDataFrame import TradingDataFrame
import calendar
from matplotlib import pyplot as plt
import seaborn as sn
import pickle

class Strategy():
    def __init__(self, name=None):
        self.name = name
        self.price_criteria_pair = None
        self.price_position_pair = None
        self.df = None

    def _calculate_maxdd(self, df):
        Roll_Max = df['daily p&l'].cummax()
        Daily_Drawdown = df['daily p&l'] - Roll_Max
        Max_Daily_Drawdown = Daily_Drawdown.cummin()
        return min(Max_Daily_Drawdown)

    def set_strategy(self, long_entry=None, long_exit=None, short_entry=None, short_exit=None):
        self.long_entry_sign, self.long_entry_threshold = long_entry
        self.long_exit_sign, self.long_exit_threshold = long_exit
        self.short_entry_sign, self.short_entry_threshold = short_entry
        self.short_exit_sign, self.short_exit_threshold = short_exit

    def apply_strategy(self, tdf):
        
        # Position long df
        if self.long_entry_sign == '>':
            df_long_entry = (tdf.criteria_df > self.long_entry_threshold) + 0

        elif self.long_entry_sign == '<':
            df_long_entry = (tdf.criteria_df < self.long_entry_threshold) + 0
        
        if self.long_exit_sign == '>':
            df_long_exit = (tdf.criteria_df > self.long_exit_threshold) * -1 + 1        # Reverse 1 and 0

        elif self.long_exit_sign == '<':
            df_long_exit = (tdf.criteria_df < self.long_exit_threshold) * -1 + 1
        
        df_long_entry = df_long_entry.replace(0, np.nan)
        df_long_exit = df_long_exit.replace(1, np.nan)

        df_position_long = df_long_entry.combine_first(df_long_exit)
        df_position_long.fillna(method='ffill', inplace=True)
        df_position_long.fillna(0, inplace=True)

        # Position short df
        if self.short_entry_sign == '>':
            df_short_entry = (tdf.criteria_df > self.short_entry_threshold) + 0

        elif self.short_entry_sign == '<':
            df_short_entry = (tdf.criteria_df < self.short_entry_threshold) + 0
        
        if self.short_exit_sign == '>':
            df_short_exit = (tdf.criteria_df > self.short_exit_threshold) * -1 + 1

        elif self.short_exit_sign == '<':
            df_short_exit = (tdf.criteria_df < self.short_exit_threshold) * -1 + 1
        
        df_short_entry = df_short_entry.replace(0, np.nan)
        df_short_exit = df_short_exit.replace(1, np.nan)

        df_position_short = df_short_entry.combine_first(df_short_exit)
        df_position_short.fillna(method='ffill', inplace=True)
        df_position_short.fillna(0, inplace=True)

        # Save position df
        df_position_long.columns = tdf.returns.columns + '_pos_long'
        df_position_short.columns = tdf.returns.columns + '_pos_short'

        self.position_long = df_position_long
        self.position_short = df_position_short
        
        print('Strategy applied!')

    def backtest(self, tdf, trading_fee=0, check_long=False, check_short=False, check_overall=True, plot=True):

        # Shift daily returns column up 1 row to calculate P&L
        returns_next_day = tdf.returns.shift(-1).copy()
        returns_next_day.columns = tdf.returns.columns + '_next_day'

        # Merge returns and positions together
        self.df = pd.merge(tdf.returns, returns_next_day, left_index=True, right_index=True)
        self.df = pd.merge(self.df, self.position_long, left_index=True, right_index=True)
        self.df = pd.merge(self.df, self.position_short, left_index=True, right_index=True)

        # Add P&L columns 
        self.df['daily p&l'] = (returns_next_day.values * (self.position_long.values - self.position_short.values)).sum(axis=1)
        self.df = self.df.iloc[1:-1,:]
        self.df['cum p&l'] = (1 + self.df['daily p&l']).cumprod() - 1

        # Other trade metrics
        self.no_of_trades = int((abs(self.position_long.diff()).sum()//2).sum() + (abs(self.position_short.diff()).sum()//2).sum())
        self.sharpe_ratio = np.sqrt(len(self.df)) * np.mean(self.df['daily p&l']) / np.std(self.df['daily p&l'])
        self.max_dd = self._calculate_maxdd(self.df)

        print(f'# of trades: {self.no_of_trades}')
        print(f'Sharpe ratio: {round(self.sharpe_ratio,2)}')
        print(f'Max Drawdown: {round(self.max_dd * 100,2)}%')

        if plot:
            plt.xticks(rotation=45)
            sn.lineplot(self.df['cum p&l'])

    def save(self):
        with open(f'.\saved_strategy\{self.name}.pickle', 'wb') as file:
            pickle.dump(self.__dict__, file)
        print(f'Saved as "{self.name}"!')

    def load(self, name=None):
        with open(f'.\saved_strategy\{name}.pickle', 'rb') as file:
            self.__dict__ = pickle.load(file)
        print(f'Loaded strategy "{self.name}"!')