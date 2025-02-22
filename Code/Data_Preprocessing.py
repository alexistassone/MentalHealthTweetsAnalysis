import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("/Users/alexistassone/Documents/AI Projects/Mental Health Tweets Analysis/Data/sentiment_tweets3.csv")

def main():
    # Display first few rows
    print(df.head())

    # Check dataset size and data types
    print(df.info())

    # Check for missing values
    print(df.isnull().sum())

    # Bar graph for label distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df["label (depression result)"], palette="coolwarm")
    plt.title("Distribution of Mental Health Labels")
    plt.xlabel("Label (O = Non-depressed, 1 = Depressed)")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    main()