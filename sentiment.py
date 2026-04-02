import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news  = stock.news
        if not news:
            return 0.0

        scores = []
        for article in news[:10]:
            # yfinance news structure has title in 'content' dict
            title = ''
            try:
                title = article.get('title', '') or \
                        article.get('content', {}).get('title', '') or \
                        article.get('content', {}).get('summary', '')
            except:
                pass
            if title:
                score = analyzer.polarity_scores(title)['compound']
                scores.append(score)

        if not scores:
            return 0.0

        avg = sum(scores) / len(scores)
        label = '🟢 Positive' if avg > 0.05 else '🔴 Negative' if avg < -0.05 else '⚪ Neutral'
        print(f"  📰 {ticker}: {label} ({avg:.2f}) from {len(scores)} headlines")
        return round(avg, 3)

    except Exception as e:
        print(f"  📰 {ticker}: Sentiment error ({e})")
        return 0.0
