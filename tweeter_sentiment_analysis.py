import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from PIL import Image


df=pd.read_csv("Dataset/twitter_dataset.csv")


df.head()


df.shape


df.info()

df.describe()


df.isnull().sum()


df.duplicated().sum()


# Remove duplicate tweets
#df = df.drop_duplicates()

# Remove rows with missing values
#df = df.dropna()

# Clean tweet text by removing special characters and URLs
df['Text'] = df['Text'].str.replace('[^a-zA-Z0-9\s]', '')
df['Text'] = df['Text'].str.replace('http\S+|www.\S+', '', case=False)


#!pip install nltk
nltk.download("punkt")

import nltk
nltk.download('stopwords')
# Tokenize tweet text
df['tokens'] = df['Text'].apply(lambda x: nltk.word_tokenize(x))
# Remove stopwords
stop_words = set(stopwords.words('english'))
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word.lower() not in stop_words])
# Stemming or Lemmatization
stemmer = PorterStemmer()
df['tokens'] = df['tokens'].apply(lambda x: [stemmer.stem(word) for word in x])


# Calculate summary statistics
mean_retweets = df['Retweets'].mean()
median_likes = df['Likes'].median()
correlation = df['Retweets'].corr(df['Likes'])
print("Mean Retweets:", mean_retweets)
print("Median Likes:", median_likes)
print("Correlation between Retweets and Likes:", correlation)
df['sentiment_polarity'] = df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity)



# plt.hist(df['sentiment_polarity'], bins=10, range=(-1, 1), edgecolor='black')
# plt.xlabel('Sentiment Polarity')
# plt.ylabel('Frequency')
# plt.title('Distribution of Sentiment Polarity in Tweets')
# plt.show()

# Perform sentiment analysis on tweet text
df['Sentiment'] = df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Categorize sentiment into positive, negative, and neutral
df['Sentiment Category'] = df['Sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

# Calculate the count of each sentiment category
sentiment_counts = df['Sentiment Category'].value_counts()

# Plot a pie chart of sentiment distribution
# plt.figure(figsize=(8, 6))
# plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
# plt.axis('equal')
# plt.title('Sentiment Distribution')
# plt.show()

# Plot the count of tweets by sentiment category
# sns.countplot(x='Sentiment Category', data=df)
# plt.xlabel('Sentiment Category')
# plt.ylabel('Count')
# plt.title('Count of Tweets by Sentiment Category')
# plt.show()

# Plot the relationship between retweets and likes
# sns.scatterplot(x='Retweets', y='Likes', data=df)
# plt.xlabel('Retweets')
# plt.ylabel('Likes')
# plt.title('Relationship between Retweets and Likes')
# plt.show()

# Plot the distribution of likes by sentiment category
# sns.boxplot(x='Sentiment Category', y='Likes', data=df)
# plt.xlabel('Sentiment Category')
# plt.ylabel('Likes')
# plt.title('Distribution of Likes by Sentiment Category')
# plt.show()

# Combine all tweet texts into a single string
all_text = ' '.join(df['Text'])

# Generate a word cloud of the most frequent words
# wordcloud = WordCloud(width=800, height=400).generate(all_text)
# plt.figure(figsize=(10, 6))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.title('Word Cloud of Most Frequent Words')
# plt.show()

# Convert the 'Timestamp' column to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Set the 'Timestamp' column as the DataFrame index
df.set_index('Timestamp', inplace=True)

# Resample the data by day and calculate the count of tweets per day
daily_tweet_count = df['Tweet_ID'].resample('D').count()

# Plot the time series of daily tweet count
# plt.figure(figsize=(12, 6))
# daily_tweet_count.plot()
# plt.xlabel('Date')
# plt.ylabel('Tweet Count')
# plt.title('Daily Tweet Count')
# plt.show()

# Combine all tweet texts into a single string
all_text = ' '.join(df['Text'])

# Split the text into individual words
words = all_text.split()

# Calculate the frequency of each word
word_counts = pd.Series(words).value_counts().sort_values(ascending=False)

#Plot the top 10 most frequent words
# plt.figure(figsize=(10, 6))
# word_counts.head(10).plot(kind='bar')
# plt.xlabel('Word')
# plt.ylabel('Frequency')
# plt.title('Top 10 Most Frequent Words')
# plt.show()

def display_top_frequent_words(df, n=10):
    # Combine all tweet texts into a single string
    all_text = ' '.join(df['Text'])

    # Split the text into individual words
    words = all_text.split()

    # Calculate the frequency of each word
    word_counts = pd.Series(words).value_counts().sort_values(ascending=False)

    # Plot the top n most frequent words
    word_counts.head(n).plot(kind='bar')
    
    print("The top word count : ",word_counts)
    return word_counts.head(n)

def top_ten_frequent_word():
    words = display_top_frequent_words(df)
    return words

# Define a function to perform analysis based on specific keyword
# def perform_analysis_based_on_keyword(df):
#     while True:
#         # Display top frequent words
#         display_top_frequent_words(df)

#         keyword = input("Enter keyword to analyze (or type 'exit' to quit): ")
#         if keyword.lower() == 'exit':
#             print("Exiting...")
#             break

#         # Filter the DataFrame based on the keyword
#         filtered_df = df[df['Text'].str.contains(keyword, case=False)]
#         print("Columns in filtered DataFrame:", filtered_df.columns)  # Debug print

#         if filtered_df.empty:
#             print("No tweets found containing the keyword:", keyword)
#             continue

#         # Perform sentiment analysis on filtered tweets
#         filtered_df['Sentiment'] = filtered_df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity)

#         # Categorize sentiment into appreciation, criticism, and neutral
#         def categorize_sentiment(score):
#             if score > 0:
#                 return 'Appreciation'
#             elif score < 0:
#                 return 'Criticism'
#             else:
#                 return 'Neutral'

#         filtered_df['Sentiment Category'] = filtered_df['Sentiment'].apply(categorize_sentiment)

#         # Calculate the count of tweets by sentiment category
#         sentiment_counts = filtered_df['Sentiment Category'].value_counts()

#         # Define colors for each sentiment category
#         colors = {'Appreciation': 'green', 'Criticism': 'red', 'Neutral': 'blue'}

#         # Plot a pie chart of sentiment distribution
#         plt.figure(figsize=(8, 6))
#         plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=[colors[s] for s in sentiment_counts.index])
#         plt.axis('equal')
#         plt.title('Sentiment Distribution for keyword: {}'.format(keyword))
#         plt.show()

#         # Plot the count of tweets by sentiment category
#         plt.figure(figsize=(10, 6))
#         sns.countplot(x='Sentiment Category', data=filtered_df, palette=colors.values())
#         plt.xlabel('Sentiment Category')
#         plt.ylabel('Count')
#         plt.title('Count of Tweets by Sentiment Category for keyword: {}'.format(keyword))
#         plt.show()

#         # Plot the relationship between retweets and likes
#         plt.figure(figsize=(10, 6))
#         sns.scatterplot(x='Retweets', y='Likes', data=filtered_df, hue='Sentiment Category', palette=colors)
#         plt.xlabel('Retweets')
#         plt.ylabel('Likes')
#         plt.title('Relationship between Retweets and Likes for keyword: {}'.format(keyword))
#         plt.legend(title='Sentiment Category')
#         plt.show()

#         # Plot the distribution of likes by sentiment category
#         plt.figure(figsize=(10, 6))
#         sns.boxplot(x='Sentiment Category', y='Likes', data=filtered_df, palette=colors.values())
#         plt.xlabel('Sentiment Category')
#         plt.ylabel('Likes')
#         plt.title('Distribution of Likes by Sentiment Category for keyword: {}'.format(keyword))
#         plt.show()

#         # Print the correlation matrix
#         correlation_matrix = filtered_df[['Retweets', 'Likes', 'Sentiment']].corr()
#         print("Correlation Matrix for keyword: {}\n".format(keyword), correlation_matrix)

#         # Generate a word cloud of the most frequent words
#         all_text_filtered = ' '.join(filtered_df['Text'])
#         wordcloud_filtered = WordCloud(width=800, height=400).generate(all_text_filtered)
#         plt.figure(figsize=(10, 6))
#         plt.imshow(wordcloud_filtered, interpolation='bilinear')
#         plt.axis('off')
#         plt.title('Word Cloud of Most Frequent Words for keyword: {}'.format(keyword))
#         plt.show()

# # Example usage:
# # Assuming df is your DataFrame containing tweets data
# perform_analysis_based_on_keyword(df)





def  analyze_sentiment(keyword):
    # Display top frequent words
    word = display_top_frequent_words(df)
    
    # Filter the DataFrame based on the keyword
    filtered_df = df[df['Text'].str.contains(keyword, case=False)]
    print("Columns in filtered DataFrame:", filtered_df.columns)  # Debug print

    if filtered_df.empty:
        print("Empty!!!")

    # Perform sentiment analysis on filtered tweets
    filtered_df['Sentiment'] = filtered_df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Categorize sentiment into appreciation, criticism, and neutral
    def categorize_sentiment(score):
        if score > 0:
            return 'Appreciation'
        elif score < 0:
            return 'Criticism'
        else:
            return 'Neutral'

        
    filtered_df['Sentiment Category'] = filtered_df['Sentiment'].apply(categorize_sentiment)

    # Calculate the count of tweets by sentiment category
    sentiment_counts = filtered_df['Sentiment Category'].value_counts()

    # Define colors for each sentiment category
    colors = {'Appreciation': 'green', 'Criticism': 'red', 'Neutral': 'blue'}

    # Plot a pie chart of sentiment distribution
    plt.figure(figsize=(8, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=[colors[s] for s in sentiment_counts.index])
    plt.axis('equal')
    plt.title('Sentiment Distribution for keyword: {}'.format(keyword))
    plt.show()

    # Plot the count of tweets by sentiment category
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Sentiment Category', data=filtered_df, palette=colors.values())
    plt.xlabel('Sentiment Category')
    plt.ylabel('Count')
    plt.title('Count of Tweets by Sentiment Category for keyword: {}'.format(keyword))
    plt.show()

    # Plot the relationship between retweets and likes
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Retweets', y='Likes', data=filtered_df, hue='Sentiment Category', palette=colors)
    plt.xlabel('Retweets')
    plt.ylabel('Likes')
    plt.title('Relationship between Retweets and Likes for keyword: {}'.format(keyword))
    plt.legend(title='Sentiment Category')
    plt.show()

    # Plot the distribution of likes by sentiment category
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Sentiment Category', y='Likes', data=filtered_df, palette=colors.values())
    plt.xlabel('Sentiment Category')
    plt.ylabel('Likes')
    plt.title('Distribution of Likes by Sentiment Category for keyword: {}'.format(keyword))
    plt.show()

    # Print the correlation matrix
    correlation_matrix = filtered_df[['Retweets', 'Likes', 'Sentiment']].corr()
    print("Correlation Matrix for keyword: {}\n".format(keyword), correlation_matrix)

    # Generate a word cloud of the most frequent words
    all_text_filtered = ' '.join(filtered_df['Text'])
    wordcloud_filtered = WordCloud(width=800, height=400).generate(all_text_filtered)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud_filtered, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Most Frequent Words for keyword: {}'.format(keyword))
    plt.show()
