# utils_module.py
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def Statistical_indicator(x_df, col):
    data = x_df[col].dropna()
    avg = data.mean()
    std = data.std()
    median = data.median()
    skew = st.skew(data)
    kurt = st.kurtosis(data)

    print(f"平均值: {avg:.4f}\n中位數: {median:.4f}\n標準差: {std:.4f}\n偏態: {skew:.4f}\n峰態: {kurt:.4f}\n")


def Data_Distribution_Plot(x_df, col):
    data = x_df[col].dropna()
    avg = data.mean()
    std = data.std()
    kde = st.gaussian_kde(data)
    x = np.linspace(avg - 3*std, avg + 3*std, 100)
    y = st.norm.pdf(x, avg, std)

    plt.figure(figsize=(18, 8))
    plt.subplot(121)
    plt.hist(data, bins=50, density=True, alpha=0.5, label='Histogram')
    plt.axvline(avg, color='red', linestyle='--', label='Mean')
    plt.axvline(avg + std, color='blue', linestyle='--', label='+1 Std')
    plt.axvline(avg - std, color='blue', linestyle='--', label='-1 Std')
    plt.legend()
    plt.title(f'{col} 分布直方圖')

    plt.subplot(122)
    plt.plot(x, kde(x), label='Kernel Density')
    plt.plot(x, y, label='Normal PDF')
    plt.axvline(avg, color='red', linestyle='--')
    plt.legend()
    plt.title(f'{col} 核密度與常態分布')
    plt.show()
