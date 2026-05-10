import yfinance as yf
from transformers import pipeline

_finbert = None

def get_finbert():
    global _finbert
    if _finbert is None:
        _finbert = pipeline('sentiment-analysis', model='ProsusAI/finbert',
                            tokenizer='ProsusAI/finbert', truncation=True, max_length=512)
    return _finbert

def get_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news  = stock.news
        if not news:
            return 0.0

        headlines = []
        for article in news[:10]:
            title = ''
            try:
                title = article.get('title', '') or \
                        article.get('content', {}).get('title', '') or \
                        article.get('content', {}).get('summary', '')
            except:
                pass
            if title:
                headlines.append(title)

        if not headlines:
            return 0.0

        results = get_finbert()(headlines, truncation=True, max_length=512, batch_size=8)
        scores = []
        for r in results:
            if r['label'] == 'positive':
                scores.append(r['score'])
            elif r['label'] == 'negative':
                scores.append(-r['score'])
            else:
                scores.append(0.0)

        avg = round(sum(scores) / len(scores), 3)
        label = '🟢 Positive' if avg > 0.05 else '🔴 Negative' if avg < -0.05 else '⚪ Neutral'
        print(f"  📰 {ticker}: {label} ({avg:.2f}) from {len(scores)} headlines")
        return avg

    except Exception as e:
        print(f"  📰 {ticker}: Sentiment error ({e})")
        return 0.0
