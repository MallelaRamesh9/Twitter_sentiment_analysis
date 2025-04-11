import pandas as pd

# Load the dataset
df = pd.read_csv("twitter_sentiment.csv")

# Display first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Check sentiment distribution
print(df['sentiment'].value_counts())

import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df['cleaned_text'] = df['text'].apply(clean_text)
print(df[['text', 'cleaned_text']].head())

from textblob import TextBlob

# Function to get sentiment polarity
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['sentiment_score'] = df['cleaned_text'].apply(get_sentiment)

# Classify sentiment into positive, neutral, negative
df['sentiment_label'] = df['sentiment_score'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

print(df['sentiment_label'].value_counts())

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=df['sentiment_label'], palette='coolwarm')
plt.title("Sentiment Distribution")
plt.show()

from wordcloud import WordCloud

text = " ".join(df['cleaned_text'])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most Common Words in Tweets")
plt.show()

df['date'] = pd.to_datetime(df['date'])  # Convert to datetime format

# Group by date and calculate average sentiment
df.groupby(df['date'].dt.date)['sentiment_score'].mean().plot(figsize=(12,5), marker='o')
plt.title("Sentiment Trends Over Time")
plt.xlabel("Date")
plt.ylabel("Average Sentiment Score")
plt.show()
