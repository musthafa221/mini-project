import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Function to load the dataset
def load_dataset():
    return pd.read_csv('Dataset/textile_product_reviews.csv')

# Function to perform sentiment analysis
def sentiment_analysis(df):
    sentiment_counts = df['sentiment'].value_counts()
    print("Sentiment Analysis:")
    print(sentiment_counts)

# Function to visualize sentiment distribution by brand
def visualize_sentiment_distribution(df):
    sentiment_distribution = df.groupby(['textile_brand', 'sentiment']).size().unstack(fill_value=0)
    sentiment_distribution.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Sentiment Analysis by Textile Brand')
    plt.xlabel('Textile Brand')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=45)
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.show()

# Function to generate word clouds for each brand
def generate_word_clouds(df):
    brand_reviews = df.groupby('textile_brand')['review'].apply(' '.join)
    for brand, reviews in brand_reviews.items():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(reviews)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud for {brand}')
        plt.axis('off')
        plt.show()

# Function to calculate top performing textile brands based on sentiment
def top_performing_brands(df):
    average_sentiment = df.groupby('textile_brand')['sentiment'].apply(lambda x: (x == 'Positive').mean())
    top_brands = average_sentiment.sort_values(ascending=False)
    print("Top performing textile brands based on sentiment:")
    print(top_brands)

# Main function to present the menu and execute the selected task
def main():
    df = load_dataset()
    while True:
        print("\nChoose an analysis task:")
        print("1. Sentiment Analysis")
        print("2. Visualize Sentiment Distribution by Brand")
        print("3. Generate Word Clouds for Each Brand")
        print("4. Calculate Top Performing Textile Brands Based on Sentiment")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            sentiment_analysis(df)
        elif choice == '2':
            visualize_sentiment_distribution(df)
        elif choice == '3':
            generate_word_clouds(df)
        elif choice == '4':
            top_performing_brands(df)
        elif choice == '5':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()
