# Categorise-Data-From-Single-Feature-Using-NLTK

## Description 
The goal was to build a product recommendation system for an application with  a feature as text based chatbot to help users find there products based on there need. Got a dataset from Kaggle but the data was messy and improper so this is what I did,
1. I preprocessed product names by extracting only nouns using NLTK, then used TF-IDF vectorization to convert them into numerical representations.
2. Used K-Means clustering to group similar product names into 20 clusters.
3. Aggregated product details (main category, subcategory, actual price) for each cluster.
4. Named each cluster based on the most common words in its product names.
5. Created a DataFrame with cluster names, product names, and related details.
6. Saved the results in 'sports.csv'.
7. Now I can train it using various algorithms such as GPT-2 , BERT , LLMA etc.

## Dataset
[Amazon Product Sales 2023](https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset)

It is a Simple Example how I Extracted Products from single feature by using nltk and clustered them in data set for future Implementation 
