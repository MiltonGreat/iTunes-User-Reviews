# iTunes-User-Reviews

### Summary and Recommendations

#### 1. Overview

This project aims to analyze user reviews from five popular social media apps—TikTok, Facebook, Instagram, YouTube, and WhatsApp—across four countries: the United States (US), France (FR), Canada (CA), and Australia (AU). The main objectives are:

- Sentiment Prediction: Predict star ratings (1-5) based on the textual content of user reviews using machine learning models such as Logistic Regression and Support Vector Classifier (SVC).
- Clustering: Group the reviews into distinct clusters using K-means clustering to identify underlying patterns in user feedback.
- Language Detection: Filter out non-English reviews for more accurate analysis.
- Text Analysis: Extract insights from clusters by analyzing the most frequent words in each cluster.

#### 2. Data

The data is collected via the iTunes API, which contains reviews for the following social media apps:

- TikTok
- Facebook
- Instagram
- YouTube
- WhatsApp

The reviews are gathered from four countries: the US, France, Canada, and Australia, each containing ratings (1-5 stars) and review text.

Data Fields:

- app_name: The name of the app being reviewed.
- country_code: The country where the review originated.
- rating: The star rating (1–5).
- review: The textual content of the review.
- sentiment_label: The derived sentiment of the review ('positive', 'neutral', 'negative') based on the rating.
- clean_review: Preprocessed review text, tokenized and lemmatized.
- cluster: The cluster assigned by K-means.
- language: The detected language of the review.

#### 3. Data Analysis Steps

1. Data Collection

    Fetching Reviews: Reviews are fetched from the iTunes API for each app and country combination.
    Extracting Reviews: The review text and rating are extracted for each entry.

2. Exploratory Data Analysis

    Distribution of Reviews: The distribution of ratings and sentiment labels is visualized across apps and countries.
    Preprocessing: The text data is tokenized, lemmatized, and cleaned to remove punctuation and convert the text to lowercase.

3. Sentiment Classification

    Label Creation: Star ratings are converted into sentiment labels:
        Ratings 1-2: Negative
        Rating 3: Neutral
        Ratings 4-5: Positive
    Text Vectorization: A TF-IDF Vectorizer is applied to convert the textual data into numerical features.
    Model Training:
        Logistic Regression and Support Vector Classifier (SVC) are trained to classify reviews based on their sentiment.

4. Clustering:

    K-means Clustering: Reviews are grouped into clusters to identify patterns, and the top 10 most common words for each cluster are displayed.
    Sentiment Distribution: The distribution of sentiments within each cluster is analyzed.

5. Language Detection:

    Non-English Review Detection: Reviews in languages other than English are detected and filtered out for more accurate analysis.
   
#### 4. Modeling Approach
1. Logistic Regression:

    TF-IDF features are used to train the model.
    The model's performance is evaluated using Precision, Recall, F1-score, and Accuracy metrics.

2. Support Vector Classifier (SVC):

    The SVC model is also trained using the TF-IDF features.
    This model is compared to Logistic Regression to assess which performs better.

3. K-means Clustering:

    K-means clustering is applied to group reviews into clusters, providing insights into the general topics or themes of the reviews..

#### 5. Key Findings
      
Sentiment Classification:

- Logistic Regression achieved an overall accuracy of 76% but struggled with class imbalance, particularly for the neutral reviews (class 3).
- Support Vector Classifier (SVC) also faced challenges with the neutral class, with lower precision and recall.
      
Sentiment Imbalance: 

- There was a significant imbalance between positive, neutral, and negative reviews, with neutral reviews being underrepresented. This led to lower precision and recall scores for neutral reviews in both Logistic Regression and SVC models.

Clustering Insights:

- Cluster 0: Contained negative reviews, where users primarily discussed frustrations with the app's features, performance, or functionality.
- Clusters 1, 2, 3, 4: Consisted of positive reviews, focusing on the app's strengths, particularly user experience, content, and social engagement.

Language Detection: 

- Non-English reviews were successfully identified and filtered out, improving the quality of sentiment analysis for English-language reviews.

#### 6. Future Work

- Addressing Class Imbalance: Use techniques such as SMOTE (Synthetic Minority Oversampling Technique) or class weighting to address the issue of imbalanced classes.
- Advanced Modeling: Explore deep learning models such as BERT or LSTM for better text classification performance.
- Multi-language Analysis: Expand the sentiment analysis to include other languages, such as French, Spanish, and Arabic.
- Topic Modeling: Use topic modeling algorithms like Latent Dirichlet Allocation (LDA) to extract more meaningful topics from the reviews.

#### 7.  Source

https://www.kaggle.com/datasets/shreyanshverma27/online-sales-dataset-popular-marketplace-data
