# Trader-Sentiment-Analysis
This project examines the interplay between Bitcoin's Fear/Greed sentiment index and trader performance on Hyperliquid. By merging sentiment data with trader metrics (e.g., PnL, win rates, leverage), it identifies patterns for optimising trading strategies. Analysis shows superior performance on "Greed" days but riskier behaviour on "Fear" days.

Trader-Sentiment-Analysis

Overview
This project analyses the relationship between Bitcoin market sentiment (Fear/Greed index) and trader behaviour/performance on Hyperliquid. The goal is to uncover patterns for smarter trading strategies, as per the Primetrade.ai Data Science Intern assignment.

Datasets
- fear_greed.csv: Daily Fear/Greed classifications (columns: Date, Classification).
- trader_data.csv: Historical trader data (columns: account, symbol, execution_price, size, side, time, start_position, event, closedPnL, leverage, etc.).

Methodology
1. Data Preparation: Load datasets, check for missing values/duplicates, convert timestamps, align by date (daily aggregation).
2. Metric Creation: Compute daily PnL per trader, win rate, trade frequency, leverage distribution, long/short ratio.
3. Analysis: Compare performance (PnL, win rate) and behaviour (trade frequency, leverage) across Fear vs. Greed days.Segment traders (e.g., high-leverage vs. low-leverage). Generate insights with charts/tables.
4. Actionable Output: Propose 2 strategy ideas based on findings.
5. Bonus: Simple predictive model for next-day PnL volatility; trader clustering.

Set Up and How to Run
1. Install dependencies: `pip install -r requirements.txt`.
2. Place datasets in `data/`.
3. Run the notebook: `jupyter notebook notebooks/analysis.ipynb`.
4. Outputs (charts, stats) are saved to `outputs/`.

Key Insights
- Traders perform better (higher PnL, win rate) on Greed days vs. Fear days.
- Behaviour shifts: Higher leverage and trade frequency on Fear days.
- Segments: High-leverage traders show more volatility; frequent traders adapt better to sentiment.

Strategy Recommendations
1. On Fear days, reduce leverage for high-leverage segments to mitigate drawdowns.
2. Increase trade frequency for frequent traders on Greed days to capitalise on momentum.

Leverage Distribution Evaluation Notes
- Data cleaning ensured no duplicates; alignment used daily grouping.
- Insights are backed by statistical tests (e.g., t-tests for PnL differences).
- Reproducible code with clear comments.
