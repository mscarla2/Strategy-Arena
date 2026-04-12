import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FINNHUB_KEY = 'd70md4hr01ql6rnvqu4gd70md4hr01ql6rnvqu50'  # Replace with your actual API key
TICKERS = ["TPET", "EONR", "USO"]
DAYS_BACK = 7               # How many days of history to fetch each run
CSV_FILE = "oil_news.csv"   # File to store all news permanently

# Keywords and weights (from Phase 2)
SCORING_WEIGHTS = {
    'war': 5.0, 'conflict': 5.0, 'iran': 5.0, 'israel': 5.0, 'hormuz': 5.0,
    'well': 2.0, 'drilling': 2.0, 'bpd': 2.0, 'production': 2.0,
    'offering': -3.0, 'dilution': -3.0, 'warrants': -3.0, 'peace': -2.0
}

# ─────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_finnhub_news(ticker, days=DAYS_BACK):
    """Fetch news from Finnhub for a given ticker."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    url = f'https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start_date}&to={end_date}&token={FINNHUB_KEY}'
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"Error fetching {ticker}: HTTP {response.status_code}")
            return []
        return response.json()
    except Exception as e:
        print(f"Exception fetching {ticker}: {e}")
        return []

def calculate_sentiment(headline):
    """Simple keyword‑based sentiment scoring."""
    score = 0
    h = headline.lower()
    for word, weight in SCORING_WEIGHTS.items():
        if word in h:
            score += weight
    return score

def load_existing_urls(csv_file):
    """Return a set of URLs already stored in the CSV file."""
    if not os.path.exists(csv_file):
        return set()
    try:
        df = pd.read_csv(csv_file)
        if 'url' in df.columns:
            return set(df['url'].dropna().tolist())
        else:
            return set()
    except Exception as e:
        print(f"Warning: Could not read existing CSV: {e}")
        return set()

def append_to_csv(new_items, csv_file):
    """Append new items to the CSV file. If file doesn't exist, create it."""
    if not new_items:
        return
    df_new = pd.DataFrame(new_items)
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_file, index=False)
    else:
        df_new.to_csv(csv_file, index=False)

# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not FINNHUB_KEY:
        print("ERROR: Please set your FINNHUB_KEY in the script.")
        return

    print(f"Fetching news for the last {DAYS_BACK} days...")
    existing_urls = load_existing_urls(CSV_FILE)
    new_articles = []

    for ticker in TICKERS:
        raw_news = get_finnhub_news(ticker)
        if not raw_news:
            continue

        for item in raw_news:
            url = item.get('url', '')
            if not url or url in existing_urls:
                continue  # skip duplicate

            # Build article record
            article = {
                'ticker': ticker,
                'date': datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d %H:%M'),
                'headline': item['headline'],
                'score': calculate_sentiment(item['headline']),
                'source': item.get('source', ''),
                'url': url,
                'fetched_at': datetime.now().isoformat()  # when we saved it
            }
            new_articles.append(article)
            existing_urls.add(url)  # avoid adding same URL later in this run

    if new_articles:
        append_to_csv(new_articles, CSV_FILE)
        print(f"Added {len(new_articles)} new articles to {CSV_FILE}.")
    else:
        print("No new articles found.")

    # Load all articles for analysis (or just use the ones we just added?)
    # For analysis, it's easier to read the whole CSV so we get the latest data.
    try:
        df_all = pd.read_csv(CSV_FILE)
    except:
        df_all = pd.DataFrame()

    if df_all.empty:
        print("No news data available.")
        return

    print("\n" + "="*70)
    print(" FINNHUB SENTIMENT ANALYSIS (all stored articles)")
    print("="*70)

    # Warn about geopolitical outliers
    war_triggers = df_all[df_all['score'] >= 5.0]
    if not war_triggers.empty:
        print(f"⚠️  WARNING: {len(war_triggers)} high‑variance 'War Premium' signals detected.")
        print("  Price action may be outlier‑driven. Use caution with mean‑reversion.\n")

    # Display top 10 most relevant news (by absolute score)
    pd.set_option('display.max_colwidth', 80)
    top_news = df_all.sort_values(by='score', key=abs, ascending=False).head(10)
    print(top_news[['ticker', 'score', 'headline']].to_string(index=False))

if __name__ == "__main__":
    main()