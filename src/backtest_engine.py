# backtest_module.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import scipy.stats as st

class BacktestEngine:
    def __init__(self, df, signal_name, method='MA'):
        self.df = df.copy()
        self.signal_name = signal_name
        self.method = method

    def generate_signal(self, col):
        df = self.df.copy()

        if self.method == 'MA':
            fast = df[col].rolling(5).mean()
            slow = df[col].rolling(20).mean()
            long_entry = (fast > slow) & (fast.shift(1) <= slow.shift(1))
            long_exit = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        elif self.method == 'BBANDS':
            mid = df[col].rolling(20).mean()
            std = df[col].rolling(20).std()
            upper = mid + 2 * std
            long_entry = (df[col] > upper) & (df[col].shift(1) <= upper.shift(1))
            long_exit = (df[col] < mid) & (df[col].shift(1) >= mid.shift(1))
        else:
            raise ValueError("Unknown method")

        signal = pd.Series(0, index=df.index)
        position = 0
        for i in range(1, len(df)):
            if long_entry.iloc[i]:
                position = 1
            elif long_exit.iloc[i]:
                position = 0
            signal.iloc[i] = position

        df['position'] = signal
        df['next_ret'] = df['pct'].shift(-1)
        df['strategy'] = df['position'] * df['next_ret']

        return df

    def plot_net_value(self):
        fig = plt.figure(figsize=(18, 8))
        ax = fig.add_subplot(111)

        for col in self.signal_name:
            df = self.df.copy()

            if self.method == 'MA':
                fast = df[col].rolling(5).mean()
                slow = df[col].rolling(20).mean()
                long_entry = (fast > slow) & (fast.shift(1) <= slow.shift(1))
                long_exit = (fast < slow) & (fast.shift(1) >= slow.shift(1))
                signal = pd.Series(0, index=df.index)
                position = 0
                for i in range(1, len(df)):
                    if long_entry.iloc[i]:
                        position = 1
                    elif long_exit.iloc[i]:
                        position = 0
                    signal.iloc[i] = position
                df['position'] = signal

            elif self.method == 'BBANDS':
                mid = df[col].rolling(20).mean()
                std = df[col].rolling(20).std()
                upper = mid + 2 * std
                lower = mid - 2 * std
                long_entry = (df[col] > upper) & (df[col].shift(1) <= upper.shift(1))
                long_exit = (df[col] < mid) & (df[col].shift(1) >= mid.shift(1))
                signal = pd.Series(0, index=df.index)
                position = 0
                for i in range(1, len(df)):
                    if long_entry.iloc[i]:
                        position = 1
                    elif long_exit.iloc[i]:
                        position = 0
                    signal.iloc[i] = position
                df['position'] = signal

            else:
                raise ValueError("Unknown method")

            df['next_ret'] = df['pct'].shift(-1)
            df['strategy'] = df['position'] * df['next_ret']
            strategy = (1 + df['strategy']).cumprod()
            ax.plot(df.index, strategy, label=f'{col}_Strategy')

        benchmark = (1 + self.df['pct']).cumprod()
        ax.plot(self.df.index, benchmark, label='Benchmark', linestyle='--', linewidth=1.5, color='black')

        ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))
        plt.xlabel('時間')
        plt.ylabel('累積報酬')
        plt.title(f'{self.method} 多策略績效')
        plt.legend()
        plt.grid()
        plt.show()

    def summary(self):
        for col in self.signal_name:
            print(f'\n=== 策略參數: {col} ===')
            df = self.generate_signal(col)
            df['pct_chg'] = df['pct']
            df['NEXT_RET'] = df['pct_chg'].shift(-1)

            RET = df['NEXT_RET'] * df['position']
            CUM_RET = (1 + RET).cumprod()

            annual_ret = CUM_RET.dropna().iloc[-1]**(250/len(RET.dropna())) - 1
            cum_ret_rate = CUM_RET.dropna().iloc[-1] - 1
            max_nv = np.maximum.accumulate(np.nan_to_num(CUM_RET))
            mdd = -np.min(CUM_RET / max_nv - 1)
            sharpe_ratio = np.mean(RET) / np.nanstd(RET, ddof=1) * np.sqrt(250)

            df['diff'] = df['position'] != df['position'].shift(1)
            df['mark_diff'] = df['diff'].cumsum()
            cond = df['position'] == 1
            temp_df = df[cond].groupby('mark_diff')['NEXT_RET'].sum()

            trade_count = len(temp_df)
            total = np.sum(df['position'])
            mean_hold = total / trade_count if trade_count != 0 else 0
            win = np.sum(RET > 0)
            lose = np.sum(RET < 0)
            win_ratio = win / total if total != 0 else 0
            mean_win_ratio = np.sum(np.where(RET > 0, RET, 0)) / win if win != 0 else 0
            mean_lose_ratio = np.sum(np.where(RET < 0, RET, 0)) / lose if lose != 0 else 0
            win_lose = win / lose if lose != 0 else np.nan

            win_count = np.sum(temp_df > 0)
            lose_count = np.sum(temp_df < 0)
            max_win = np.max(temp_df)
            max_lose = np.min(temp_df)
            win_rat = win_count / trade_count if trade_count != 0 else 0
            mean_win = np.sum(np.where(temp_df > 0, temp_df, 0)) / trade_count if trade_count != 0 else 0
            mean_lose = np.sum(np.where(temp_df < 0, temp_df, 0)) / trade_count if trade_count != 0 else 0
            mean_win_lose = win_count / lose_count if lose_count != 0 else np.nan

            print(f"年化報酬率: {annual_ret:.2f}%")
            print(f"累積報酬率: {cum_ret_rate:.2f}%")
            print(f"夏普比率: {sharpe_ratio:.2f}")
            print(f"最大回撤: {mdd:.2f}%")
            print(f"持倉總天數: {total}")
            print(f"交易次數: {trade_count}")
            print(f"平均持倉天數: {mean_hold:.2f}")
            print(f"獲利天數: {win}")
            print(f"虧損天數: {lose}")
            print(f"勝率(按天): {win_ratio:.2f}%")
            print(f"平均獲利率(按天): {mean_win_ratio:.2f}%")
            print(f"平均虧損率(按天): {mean_lose_ratio:.2f}%")
            print(f"平均盈虧比(按天): {win_lose:.2f}")
            print(f"獲利次數: {win_count}")
            print(f"虧損次數: {lose_count}")
            print(f"單次最大獲利: {max_win:.2f}%")
            print(f"單次最大虧損: {max_lose:.2f}%")
            print(f"勝率(按次): {win_rat:.2f}%")
            print(f"平均獲利率(按次): {mean_win:.2f}%")
            print(f"平均虧損率(按次): {mean_lose:.2f}%")
            print(f"平均盈虧比(按次): {mean_win_lose:.2f}")
        
def Statistical_indicator(df, col):
    data = df[col].dropna()
    avgRet = np.mean(data)
    medianRet = np.median(data)
    stdRet = np.std(data)
    skewRet = st.skew(data)
    kurtRet = st.kurtosis(data)

    print("""
平均數 : {:.4f}
中位數 : {:.4f}
標準差 : {:.4f}
偏態   : {:.4f}
峰度   : {:.4f}
+1 標準差 : {:.4f}
-1 標準差 : {:.4f}
最小值 : {:.4f}
最大值 : {:.4f}
""".format(avgRet, medianRet, stdRet, skewRet, kurtRet, avgRet+stdRet, avgRet-stdRet, min(data), max(data)))


def Data_Distribution_Plot(df, col):
    data = df[col].dropna()
    indicator = Statistical_indicator(df, col)

    avg = np.mean(data)
    std = np.std(data)

    x = np.linspace(avg - 3 * std, avg + 3 * std, 100)
    y = st.norm.pdf(x, avg, std)
    kde = st.gaussian_kde(data)
    
    fig = plt.figure(figsize=(18, 9))
    plt.subplot(121)
    plt.hist(data, 50, weights=np.ones(len(data)) / len(data), alpha=0.4)
    plt.axvline(avg, color='red', linestyle='--', linewidth=0.8, label='平均')
    plt.axvline(avg - std, color='blue', linestyle='--', linewidth=0.8, label='-1 標準差')
    plt.axvline(avg + std, color='blue', linestyle='--', linewidth=0.8, label='+1 標準差')
    plt.ylabel('比例')
    plt.legend()

    plt.subplot(122)
    plt.plot(x, kde(x), label='Kernel Density')
    plt.plot(x, y, color='black', linewidth=1, label='Normal Fit')
    plt.axvline(avg, color='red', linestyle='--', linewidth=0.8, label='平均')
    plt.legend()
    plt.show()

def add_all_kernel(df, periods):
    for n in periods:
        turnover_ma = df['turnover_rate_f'].rolling(n).mean()
        std = df['pct'].rolling(n).std(ddof=0)
        df[f'kernel_{n}'] = std / turnover_ma
    return df