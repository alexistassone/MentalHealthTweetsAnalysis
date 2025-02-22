import Data_Preprocessing as dp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Generate wordclouds
def generate_wordcloud(text_data, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(' '.join(text_data))
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=14)
    plt.show()

def main():
    # Wordcloud for depressed tweets
    generate_wordcloud(dp.df[dp.df["label (depression result)"]==1]["message to examine"], "Most Common Words in Depressed Tweets")

    # Wordcloud for non-depressed tweets
    generate_wordcloud(dp.df[dp.df["label (depression result)"]==0]["message to examine"], "Most Common Words in Non-Depressed Tweets")

if __name__ == "__main__":
    main()