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

    def _calculate_daily_pl(self, row):
        pl = 0
        for price, positions in self.price_position_pair.items():
            pl += row[price+'_pct_change_next_day'] * (row[positions[0]] - row[positions[1]])    # Long -> *1, short -> *-1
        return pl
    
    def _calculate_maxdd(self, tdf, window):
        return (self.df['cum p&l'].rolling(window, min_periods=1).max() - self.df['cum p&l'].rolling(window, min_periods=1).min()).max()

    def set_strategy(self, long_entry=None, long_exit=None, short_entry=None, short_exit=None):
        self.long_entry_sign, self.long_entry_threshold = long_entry
        self.long_exit_sign, self.long_exit_threshold = long_exit
        self.short_entry_sign, self.short_entry_threshold = short_entry
        self.short_exit_sign, self.short_exit_threshold = short_exit

    def apply_strategy(self, tdf):
        
        self.df = tdf.data.copy()
        self.price_position_pair = {}
        for price, criteria in tdf.price_criteria_pair.items():
            if self.long_entry_sign == '>':
                self.df.loc[self.df[criteria] > self.long_entry_threshold, price+'_Position_long'] = 1
            elif self.long_entry_sign == '<':
                self.df.loc[self.df[criteria] < self.long_entry_threshold, price+'_Position_long'] = 1
            else:
                print('error in long entry sign')

            if self.long_exit_sign == '>':
                self.df.loc[self.df[criteria] > self.long_exit_threshold, price+'_Position_long'] = 0
            elif self.long_exit_sign == '<':
                self.df.loc[self.df[criteria] < self.long_exit_threshold, price+'_Position_long'] = 0
            else:
                print('error in long exit sign')
            
            if self.short_entry_sign == '>':
                self.df.loc[self.df[criteria] > self.short_entry_threshold, price+'_Position_short'] = 1
            elif self.short_entry_sign == '<':
                self.df.loc[self.df[criteria] < self.short_entry_threshold, price+'_Position_short'] = 1
            else:
                print('error in short entry sign')
            
            if self.short_exit_sign == '>':
                self.df.loc[self.df[criteria] > self.short_exit_threshold, price+'_Position_short'] = 0
            elif self.short_exit_sign == '<':
                self.df.loc[self.df[criteria] < self.short_exit_threshold, price+'_Position_short'] = 0
            else:
                print('error in short exit sign')

            self.price_position_pair[price] = [price+'_Position_long', price+'_Position_short']

        self.df.fillna(method='ffill', inplace=True)
        self.df.fillna(0, inplace=True)
        print('Strategy applied!')

    def backtest(self, tdf, trading_fee=0, check_long=False, check_short=False, check_overall=True, plot=True):

        # Shift daily returns column up 1 row to calculate P&L
        for price in tdf.price_criteria_pair.keys():
            self.df[price+'_pct_change_next_day'] = self.df[price+'_pct_change'].shift(-1)

        self.df = self.df.iloc[1:-1,:]

        # Add P&L columns 
        self.df['daily p&l'] = self.df.apply(self._calculate_daily_pl, axis=1)
        self.df['cum p&l'] = (1 + self.df['daily p&l']).cumprod() - 1

        # Other trade metrics
        self.no_of_trades = abs(self.df['Open_Position_long'].astype(int) - self.df['Open_Position_long'].shift().fillna(0).astype(int)).sum()//2
        self.sharpe_ratio = np.sqrt(len(self.df)) * np.mean(self.df['daily p&l']) / np.std(self.df['daily p&l'])
        self.max_dd = self._calculate_maxdd(tdf, window=len(self.df))
        #self.max_dd = (self.df['cum p&l'].rolling(len(self.df), min_periods=1).max() - self.df['cum p&l'].rolling(len(self.df), min_periods=1).min()).max()

        print(f'# of trades: {self.no_of_trades}')
        print(f'Sharpe ratio: {round(self.sharpe_ratio,2)}')
        print(f'Max Drawdown: {self.max_dd}')

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