# -*- coding: utf-8 -*-
"""Spam_or_Ham_classifier_nlp.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ltvSqlBEzKNpC84g8DN6KsvTyloQgMeF

## Import Dependencies
"""

import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

"""## Reading the SMS spam dataset into a pandas DataFrame"""

df = pd.read_csv('/content/SMSSpamCollection.csv', sep='\t',
                           names=["label", "message"])
df.head()

"""## Displaying information about the DataFrame, including data types and non-null counts"""

df.info()

"""## Checking for null values in the DataFrame"""

df.isnull().sum()

"""## Displaying the count of each unique value in the 'label' column"""

df['label'].value_counts()

"""## Initializing the Porter Stemmer and creating a corpus of processed text data"""

ps = PorterStemmer()
corpus = []
for i in range(0, len(df)):
  # Removing non-alphabetic characters and converting text to lowercase
    review = re.sub('[^a-zA-Z]', ' ', df['message'][i])
    review = review.lower()
    review = review.split()

    # Stemming and removing stopwords from the text
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

"""## Creating a bag-of-words model using CountVectorizer with a maximum of 2500 features"""

cv = CountVectorizer(max_features=2500)
x = cv.fit_transform(corpus).toarray()

# Creating dummy variables for the 'label' column and extracting the 'spam' column
y=pd.get_dummies(df['label'])
y=y.iloc[:,1].values

"""## Splitting the dataset into training and testing sets"""

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

"""## Initializing and training a Multinomial Naive Bayes model for spam detection"""

spam_detect_model = MultinomialNB()
spam_detect_model.fit(x_train, y_train)

"""## Making predictions on the test set"""

y_pred=spam_detect_model.predict(x_test)

"""## Evaluating the model performance using confusion matrix and accuracy score"""

from sklearn.metrics import confusion_matrix, accuracy_score

result = confusion_matrix(y_test,y_pred)
result

final_accuracy = accuracy_score(y_test,y_pred)
final_accuracy

"""## Making a prediction for a new SMS and displaying the result"""

new_email = "Please reply to get this offer"
new_email_transformed = cv.transform([new_email])

new_email_pred = spam_detect_model.predict(new_email_transformed)
print("Class label prediction:", new_email_pred[0])

if new_email_pred == 1:
  print("Ham")
else:
  print("spam")
