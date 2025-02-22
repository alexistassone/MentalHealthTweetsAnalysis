import Data_Preprocessing as dp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Initialize snetiment analyzer
sia = SentimentIntensityAnalyzer()

# Apply sentiment analysis
dp.df["sentiment_score"] = dp.df["message to examine"].apply(lambda tweet: sia.polarity_scores(tweet)["compound"])

def main():
    # Visualize sentiment scores
    plt.figure(figsize=(8, 5))
    sns.histplot(dp.df["sentiment_score"], bins=30, kde=True, color="purple")
    plt.title("Sentiment Scores in Tweets")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Density")
    plt.show()

if __name__ == "__main__":
    main()