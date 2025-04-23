import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import scipy.stats as st


class VT_Factor:
    def __init__(self, df, periods=250, forward_days=[5, 10, 15, 20]):
        self.df = df.copy()
        self.periods = periods if isinstance(periods, list) else [periods]
        self.forward_days = forward_days

    def _calc_kernel(self, df, period):
        turnover_ma = df['turnover_rate_f'].rolling(period).mean()
        std = df['pct'].rolling(period).std(ddof=0)
        return (std / turnover_ma).rename(f'kernel_{period}')

    def get_signal(self):
        df = self.df.copy()
        for p in self.periods:
            df[f'kernel_{p}'] = self._calc_kernel(df, p)

        for n in self.forward_days:
            df[f'future_{n}d_ret'] = df['close'].pct_change(n).shift(-n)

        return df

    def forward_distribution_plot(self, period, bins=50):
        df = self.get_signal()
        signal_col = f'kernel_{period}'
        df = df[[signal_col] + [f'future_{n}d_ret' for n in self.forward_days]].dropna()
        df['bin'] = pd.cut(df[signal_col], bins=bins)

        grouped = df.groupby('bin')

        fig, axs = plt.subplots(len(self.forward_days), 1, figsize=(18, 5 * len(self.forward_days)))
        for i, n in enumerate(self.forward_days):
            mean_ret = grouped[f'future_{n}d_ret'].mean()
            axs[i].bar(mean_ret.index.astype(str), mean_ret.values)
            axs[i].set_title(f'{n}-day Forward Return by Signal Bin')
            axs[i].tick_params(axis='x', rotation=90)
        plt.tight_layout()
        plt.show()

    def QuantReg_plot(self, n=15, forward_col=None):
        df = self.get_signal()
        period = self.periods[0]
        signal_col = f'kernel_{period}'
        if forward_col is None:
            forward_col = f'future_{n}d_ret'

        df = df[[signal_col, forward_col]].dropna()
        model = smf.quantreg(f'{forward_col} ~ {signal_col}', df)

        quantiles = np.arange(0.05, 0.96, 0.1)
        models = []
        for q in quantiles:
            res = model.fit(q=q)
            params = res.params[signal_col]
            ci = res.conf_int().loc[signal_col]
            models.append([q, params, ci[0], ci[1]])

        models_df = pd.DataFrame(models, columns=['quantile', 'beta', 'lower', 'upper'])
        plt.plot(models_df['quantile'], models_df['beta'], label='Quantile Beta', color='red')
        plt.fill_between(models_df['quantile'], models_df['lower'], models_df['upper'], color='gray', alpha=0.3)
        plt.axhline(y=0, linestyle='--', color='black')
        plt.xlabel('Quantile')
        plt.ylabel('Beta')
        plt.title(f'Quantile Regression of {forward_col} on {signal_col}')
        plt.legend()
        plt.show()

    def summary(self):
        df = self.get_signal()
        # df['pct_chg'] = df['pct_chg'] / 100
        df['next_ret'] = df['pct'].shift(-1)

        signal_cols = [f'kernel_{p}' for p in self.periods]
        result = {}

        for col in signal_cols:
            fast = df[col].rolling(5).mean()
            slow = df[col].rolling(20).mean()
            signal = (fast > slow) & (fast.shift(1) <= slow.shift(1))
            position = signal.replace(False, np.nan).ffill().fillna(False).astype(int)

            ret = df['next_ret'] * position
            cum_ret = (1 + ret).cumprod() - 1
            annual_ret = (1 + cum_ret.iloc[-1]) ** (250 / len(ret.dropna())) - 1 if len(ret.dropna()) > 0 else 0
            sharpe = ret.mean() / ret.std() * np.sqrt(250) if ret.std() > 0 else 0
            drawdown = (1 + ret).cumprod() / (1 + ret).cumprod().cummax() - 1
            max_dd = drawdown.min()

            result[col] = {
                '年化報酬率': f'{annual_ret:.2%}',
                '累積報酬率': f'{cum_ret.iloc[-1]:.2%}',
                '夏普比率': f'{sharpe:.2f}',
                '最大回撤': f'{max_dd:.2%}'
            }

        return pd.DataFrame(result).T