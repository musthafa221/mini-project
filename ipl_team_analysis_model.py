import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import re
import nltk
from ipywidgets import interact, widgets

# Download NLTK resources
nltk.download('vader_lexicon')

# Load the corrected dataset
df = pd.read_csv('Dataset/ipldataset.csv')

# Function to extract hashtags from tweets
def extract_hashtags(tweet):
    hashtags = re.findall(r'#(\w+)', tweet)
    return hashtags

# Apply function to extract hashtags from tweets
df['hashtags'] = df['tweets'].apply(extract_hashtags)

# Group by team and count the occurrence of each hashtag
hashtag_counts = df.explode('hashtags').groupby('team')['hashtags'].value_counts().reset_index(name='count')

# Identify popular hashtags associated with each team
popular_hashtags = hashtag_counts.groupby('team').head(5)

# Function to perform sentiment analysis
def perform_sentiment_analysis():
    sid = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['tweets'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['sentiment'] = df['sentiment_score'].apply(lambda x: 'Positive' if x > 0.05 else 'Negative' if x < -0.05 else 'Neutral')
    sentiment_distribution = df.groupby(['team', 'sentiment']).size().unstack(fill_value=0)
    sentiment_distribution.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Sentiment Analysis of Tweets for Each Team')
    plt.xlabel('Team')
    plt.ylabel('Number of Tweets')
    plt.xticks(rotation=45)
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.show()

# Function to perform engagement analysis
def perform_engagement_analysis():
    total_engagement = df.groupby('team').agg({'retweets': 'sum', 'likes': 'sum'}).reset_index()
    total_engagement = total_engagement.sort_values(by=['retweets', 'likes'], ascending=False)
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    axes[0].bar(total_engagement['team'], total_engagement['retweets'], color='skyblue')
    axes[0].set_title('Total Retweets by Team')
    axes[0].set_xlabel('Team')
    axes[0].set_ylabel('Total Retweets')
    axes[0].tick_params(axis='x', rotation=45)
    axes[1].bar(total_engagement['team'], total_engagement['likes'], color='lightcoral')
    axes[1].set_title('Total Likes by Team')
    axes[1].set_xlabel('Team')
    axes[1].set_ylabel('Total Likes')
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

# Function to perform tweet activity analysis
def perform_tweet_activity_analysis():
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    tweet_activity = df.groupby(['team', 'date']).size().reset_index(name='tweet_count')
    fig, ax = plt.subplots(figsize=(12, 6))
    for team in df['team'].unique():
        team_data = tweet_activity[tweet_activity['team'] == team]
        ax.plot(team_data['date'], team_data['tweet_count'], label=team)
    ax.set_title('Tweet Activity Over Time for Each Team')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Tweets')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to perform word cloud analysis
def perform_word_cloud_analysis():
    for team in df['team'].unique():
        team_tweets = ' '.join(df[df['team'] == team]['tweets'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(team_tweets)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud for {team}')
        plt.axis('off')
        plt.show()

# Function to perform hashtag analysis
def perform_hashtag_analysis():
    def extract_hashtags(tweet):
        hashtags = re.findall(r'#(\w+)', tweet)
        return hashtags
    df['hashtags'] = df['tweets'].apply(extract_hashtags)
    hashtag_counts = df.explode('hashtags').groupby('team')['hashtags'].value_counts().reset_index(name='count')
    popular_hashtags = hashtag_counts.groupby('team').head(5)
    print("Popular hashtags associated with each team:")
    print(popular_hashtags)

# Define function to handle analysis selection
def select_analysis(analysis):
    if analysis == 'Sentiment Analysis':
        perform_sentiment_analysis()
    elif analysis == 'Engagement Analysis':
        perform_engagement_analysis()
    elif analysis == 'Tweet Activity Analysis':
        perform_tweet_activity_analysis()
    elif analysis == 'Word Cloud Analysis':
        perform_word_cloud_analysis()
    elif analysis == 'Hashtag Analysis':
        perform_hashtag_analysis()

