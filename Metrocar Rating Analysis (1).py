#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas nltk


# In[2]:


import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


nltk.download('vader_lexicon')

# Load Metrocar reviews data 
reviews_df = pd.read_csv(r'C:\Users\cecil\Downloads\reviewsdata.csv')


# In[3]:


# Sentiment Intensity Analyzer
sid = SentimentIntensityAnalyzer()

# Get sentiment of a text
def get_sentiment(text):
    scores = sid.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis to the 'review' column and create a new column 'sentiment'
reviews_df['sentiment'] = reviews_df['review'].apply(get_sentiment)


# In[4]:


# Analyze sentiment distribution
sentiment_counts = reviews_df['sentiment'].value_counts()


print(sentiment_counts)

# Visualize sentiment distribution using matplotlib
import matplotlib.pyplot as plt

plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.title('Sentiment Distribution of Metrocar Reviews')
plt.show()


# In[5]:


negative_reviews = reviews_df[reviews_df['sentiment'] == 'Negative']


# In[6]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Download NLTK resources 
nltk.download('punkt')
nltk.download('stopwords')

# Tokenize the negative reviews and remove stopwords
stop_words = set(stopwords.words('english'))

negative_reviews_tokens = []
for review in negative_reviews['review']:
    words = word_tokenize(review.lower())  # Convert to lowercase
    words = [word for word in words if word.isalpha()]  # Remove punctuation
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    negative_reviews_tokens.extend(words)


# In[7]:


word_counts = Counter(negative_reviews_tokens)
print(word_counts.most_common(50))


# In[8]:


pip install wordcloud


# In[9]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[10]:


# Extract words and their frequencies for the bar chart
words, frequencies = zip(*word_counts.most_common(10))

plt.figure(figsize=(12, 6))
plt.bar(words, frequencies)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.title('Most Common Words in Negative Reviews')
plt.show()


# In[11]:


positive_reviews = reviews_df[reviews_df['sentiment'] == 'Positive']


# In[12]:


positive_reviews_tokens = []
for review in positive_reviews['review']:
    words = word_tokenize(review.lower())  # Convert to lowercase
    words = [word for word in words if word.isalpha()]  # Remove punctuation
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    positive_reviews_tokens.extend(words)


# In[13]:


word_counts = Counter(positive_reviews_tokens)
print(word_counts.most_common(50))


# In[14]:


# Extract words and their frequencies for bar chart
words, frequencies = zip(*word_counts.most_common(10))

plt.figure(figsize=(12, 6))
plt.bar(words, frequencies)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.title('Most Common Words in Positive Reviews')
plt.show()


# In[15]:


# Analyzing User Ratings
average_rating = reviews_df['rating'].mean()  # Calculate average rating
rating_counts = reviews_df['rating'].value_counts()  # Count the occurrences of each rating

# Print the results
print("Average Rating:", average_rating)
print("Rating Counts:")
print(rating_counts)


# In[16]:


ratings_counts = reviews_df['rating'].value_counts().sort_index()

# Plotting the distribution of ratings
plt.figure(figsize=(8, 6))
plt.bar(ratings_counts.index, ratings_counts.values)
plt.xlabel('Ratings')
plt.ylabel('Count')
plt.title('Distribution of Ratings')
plt.xticks(ratings_counts.index)  # Set x-ticks to the unique ratings
plt.show()


# In[ ]:





# In[ ]:




