# plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# Add plotting functions here 

def plot_volatility_smile(df):
    """
    Plot volatility smile analysis by time to expiration.
    """
    df = df.copy()
    df['tte_bin'] = pd.cut(df['tte'], bins=[0, 14, 30, 45, 60], labels=['0-14d', '14-30d', '30-45d', '45-60d'])
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Volatility Smile Analysis by Time to Expiration', fontsize=16)
    colors = ['blue', 'green', 'red', 'orange']
    tte_bins = df['tte_bin'].dropna().unique()
    for i, (ax, tte_bin) in enumerate(zip(axes.flat, tte_bins)):
        if pd.isna(tte_bin):
            continue
        subset = df[df['tte_bin'] == tte_bin]
        if len(subset) > 0 and 'moneyness' in subset.columns:
            subset['moneyness_bin'] = pd.cut(subset['moneyness'], bins=20)
            smile_data = subset.groupby('moneyness_bin')['impl_volatility'].agg(['mean', 'count']).reset_index()
            smile_data = smile_data[smile_data['count'] >= 5]
            if len(smile_data) > 0:
                moneyness_centers = smile_data['moneyness_bin'].apply(lambda x: x.mid)
                ax.plot(moneyness_centers, smile_data['mean'], 'o-', color=colors[i % len(colors)])
                ax.set_xlabel('Moneyness')
                ax.set_ylabel('Implied Volatility')
                ax.set_title(f'TTE: {tte_bin}')
                ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_term_structure(term_structure, atm_options):
    """
    Plot implied volatility term structure and scatter with trend.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].errorbar(term_structure['tte'], term_structure['mean'], yerr=term_structure['std'], marker='o', capsize=5, capthick=2)
    axes[0].set_xlabel('Time to Expiration (Days)')
    axes[0].set_ylabel('Implied Volatility')
    axes[0].set_title('Implied Volatility Term Structure (ATM Options)')
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(atm_options['tte'], atm_options['impl_volatility'], alpha=0.5)
    if len(term_structure) > 2:
        z = np.polyfit(term_structure['tte'], term_structure['mean'], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(term_structure['tte'].min(), term_structure['tte'].max(), 100)
        axes[1].plot(x_trend, p(x_trend), "r--", alpha=0.8, label='Quadratic Fit')
        axes[1].legend()
    axes[1].set_xlabel('Time to Expiration (Days)')
    axes[1].set_ylabel('Implied Volatility')
    axes[1].set_title('IV vs TTE Scatter with Trend')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_event_study_iv(event_df):
    """
    Plot pre- and post-earnings implied volatility distributions.
    """
    if event_df is None or event_df.empty:
        print("No event study data to plot.")
        return
    pre = event_df[event_df['period'] == 'pre_earnings']['impl_volatility']
    post = event_df[event_df['period'] == 'post_earnings']['impl_volatility']
    plt.figure(figsize=(10, 6))
    plt.hist(pre, bins=30, alpha=0.6, label='Pre-Earnings IV')
    plt.hist(post, bins=30, alpha=0.6, label='Post-Earnings IV')
    plt.title('Pre vs Post Earnings Implied Volatility')
    plt.xlabel('Implied Volatility')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show() 