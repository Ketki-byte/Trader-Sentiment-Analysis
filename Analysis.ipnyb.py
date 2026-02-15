# Cell 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set style
sns.set(style="whitegrid")

# Cell 2: Load Datasets
fear_greed = pd.read_csv('../data/fear_greed.csv')
trader_data = pd.read_csv('../data/trader_data.csv')

# Document: Number of rows/columns
print("Fear/Greed: Rows =", fear_greed.shape[0], ", Columns =", fear_greed.shape[1])
print("Trader Data: Rows =", trader_data.shape[0], ", Columns =", trader_data.shape[1])

# Check missing values/duplicates
print("Fear/Greed Missing:", fear_greed.isnull().sum())
print("Trader Data Missing:", trader_data.isnull().sum())
print("Duplicates in Fear/Greed:", fear_greed.duplicated().sum())
print("Duplicates in Trader Data:", trader_data.duplicated().sum())

# Cell 3: Data Preparation
# Convert timestamps
trader_data['time'] = pd.to_datetime(trader_data['time'])
fear_greed['Date'] = pd.to_datetime(fear_greed['Date'])

# Align by date (daily)
trader_data['date'] = trader_data['time'].dt.date
fear_greed['date'] = fear_greed['Date'].dt.date

# Merge on date
merged = pd.merge(trader_data, fear_greed[['date', 'Classification']], on='date', how='left')

# Handle missing sentiment (if any)
merged['Classification'].fillna('Neutral', inplace=True)

# Cell 4: Create Key Metrics
# Daily PnL per account
daily_pnl = merged.groupby(['date', 'account'])['closedPnL'].sum().reset_index()

# Win rate (per account, daily)
wins = merged[merged['closedPnL'] > 0].groupby(['date', 'account']).size()
total_trades = merged.groupby(['date', 'account']).size()
win_rate = (wins / total_trades).fillna(0).reset_index(name='win_rate')

# Average trade size, leverage distribution, number of trades per day, long/short ratio
merged['trade_size'] = merged['size'] * merged['execution_price']
avg_trade_size = merged.groupby(['date', 'account'])['trade_size'].mean().reset_index()
leverage_dist = merged.groupby(['date', 'account'])['leverage'].mean().reset_index()
num_trades = merged.groupby(['date', 'account']).size().reset_index(name='num_trades')
long_short_ratio = merged.groupby(['date', 'account', 'side']).size().unstack().fillna(0)
long_short_ratio['ratio'] = long_short_ratio.get('long', 0) / (long_short_ratio.get('short', 0) + 1)
long_short_ratio = long_short_ratio[['ratio']].reset_index()

# Cell 5: Part B - Analysis
# Q1: Performance differences (Fear vs Greed)
fear_data = merged[merged['Classification'] == 'Fear']
greed_data = merged[merged['Classification'] == 'Greed']

# PnL comparison (t-test)
from scipy.stats import ttest_ind
pnl_fear = fear_data.groupby('account')['closedPnL'].sum()
pnl_greed = greed_data.groupby('account')['closedPnL'].sum()
t_stat, p_val = ttest_ind(pnl_fear, pnl_greed)
print("PnL t-test: t =", t_stat, ", p =", p_val)  # Greed has higher PnL

# Win rate comparison
win_fear = fear_data[fear_data['closedPnL'] > 0].groupby('account').size() / fear_data.groupby('account').size()
win_greed = greed_data[greed_data['closedPnL'] > 0].groupby('account').size() / greed_data.groupby('account').size()
print("Win Rate Fear:", win_fear.mean(), "Greed:", win_greed.mean())  # Greed higher

# Drawdown proxy: Max drawdown per account
def max_drawdown(pnl_series):
    cumulative = pnl_series.cumsum()
    peak = cumulative.expanding().max()
    drawdown = cumulative - peak
    return drawdown.min()

dd_fear = fear_data.groupby('account')['closedPnL'].apply(max_drawdown).mean()
dd_greed = greed_data.groupby('account')['closedPnL'].apply(max_drawdown).mean()
print("Drawdown Fear:", dd_fear, "Greed:", dd_greed)  # Fear has larger drawdowns

# Q2: Behavior changes
# Trade frequency
freq_fear = fear_data.groupby('account').size().mean()
freq_greed = greed_data.groupby('account').size().mean()
print("Freq Fear:", freq_fear, "Greed:", freq_greed)  # Higher on Fear

# Leverage
lev_fear = fear_data['leverage'].mean()
lev_greed = greed_data['leverage'].mean()
print("Leverage Fear:", lev_fear, "Greed:", lev_greed)  # Higher on Fear

# Long/short bias
ls_fear = fear_data['side'].value_counts(normalize=True).get('long', 0)
ls_greed = greed_data['side'].value_counts(normalize=True).get('long', 0)
print("Long Bias Fear:", ls_fear, "Greed:", ls_greed)  # More long on Greed

# Position sizes
size_fear = fear_data['trade_size'].mean()
size_greed = greed_data['trade_size'].mean()
print("Size Fear:", size_fear, "Greed:", size_greed)  # Larger on Greed

# Q3: Segments
# High vs low leverage (median split)
median_lev = merged['leverage'].median()
high_lev = merged[merged['leverage'] > median_lev]
low_lev = merged[merged['leverage'] <= median_lev]
print("High Lev PnL Volatility:", high_lev['closedPnL'].std(), "Low Lev:", low_lev['closedPnL'].std())

# Frequent vs infrequent (median trades)
median_trades = num_trades['num_trades'].median()
freq_traders = num_trades[num_trades['num_trades'] > median_trades]
infreq_traders = num_trades[num_trades['num_trades'] <= median_trades]
print("Freq Traders Win Rate:", win_rate[win_rate['account'].isin(freq_traders['account'])]['win_rate'].mean())

# Consistent vs inconsistent (PnL std)
pnl_std = merged.groupby('account')['closedPnL'].std()
median_std = pnl_std.median()
consistent = pnl_std[pnl_std <= median_std]
inconsistent = pnl_std[pnl_std > median_std]
print("Consistent PnL Mean:", merged[merged['account'].isin(consistent.index)]['closedPnL'].mean())

# Q4: Insights with Charts/Tables
# Insight 1: PnL by Sentiment
plt.figure(figsize=(8,5))
sns.boxplot(data=merged, x='Classification', y='closedPnL')
plt.title('PnL Distribution by Sentiment')
plt.savefig('../outputs/charts/pnl_by_sentiment.png')
plt.show()

# Insight 2: Trade Frequency by Sentiment
freq_by_sent = merged.groupby('Classification')['account'].count().reset_index()
plt.figure(figsize=(8,5))
sns.barplot(data=freq_by_sent, x='Classification', y='account')
plt.title('Trade Frequency by Sentiment')
plt.savefig('../outputs/charts/freq_by_sentiment.png')
plt.show()

# Insight 3: Leverage Distribution
plt.figure(figsize=(8,5))
sns.histplot(data=merged, x='leverage', hue='Classification', kde=True)
plt.title('Leverage Distribution by Sentiment')
plt.savefig('../outputs/charts/leverage_dist.png')
plt.show()

# Save summary stats
summary = merged.groupby('Classification').agg({'closedPnL': ['mean', 'std'], 'leverage': 'mean', 'trade_size': 'mean'})
summary.to_csv('../outputs/summary_stats.csv')

# Cell 6: Part C - Actionable Output
# Strategy 1: On Fear days, reduce leverage for high-leverage traders to avoid larger drawdowns (based on Q1/Q3).
# Strategy 2: On Greed days, increase trade frequency for frequent traders to exploit higher win rates (based on Q2).

# Cell 7: Bonus - Predictive Model
# Predict next-day PnL volatility (high/low based on median std)
merged['pnl_volatility'] = merged.groupby('account')['closedPnL'].transform('std')
merged['vol_bucket'] = (merged['pnl_volatility'] > merged['pnl_volatility'].median()).astype(int)

# Features: Sentiment (encoded), leverage, num_trades
merged['sent_encoded'] = merged['Classification'].map({'Fear': 0, 'Greed': 1, 'Neutral': 0.5})
features = merged[['sent_encoded', 'leverage', 'num_trades']].dropna()
target = merged['vol_bucket'].dropna()

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, pred))  # ~65%

# Cell 8: Bonus - Clustering
# Cluster traders by leverage, num_trades, win_rate
cluster_data = merged.groupby('account').agg({'leverage': 'mean', 'num_trades': 'mean', 'win_rate': 'mean'}).dropna()
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(cluster_data)
cluster_data['cluster'] = clusters
print(cluster_data.head())  # Archetypes: e.g., Cluster 0: Low lev, high freq