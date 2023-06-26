import numpy as np
import pandas as pd
import re
import os

import nltk
from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from transformers import pipeline


# Use pretrained model from https://huggingface.co/spaces/yangheng/PyABSA for Aspect Based Sentiment Analysis
absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
absa_model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")

#Inital setting for Tokenize text & Lemmatize text & Remove stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Concepts with similar aspects (Customed Aspects)
concepts = {
    'environment': ['interior', 'table', 'ambient', 'atmosphere', 'washroom', 'kitchen'],
    'server': ['staff', 'employee', 'server', 'attitude','waitress', 'waiter','worker',' job', 'tip'],
    'service': ['service', 'reservation', 'order', 'time', 'schedule'],
    'line': ['wait', 'line', 'seat'],
    'food': ['menu', 'food', 'dish' 'appetizer', 'course', 'size', 'bowl', 'variety', 'choice', 'piece'],
    'dessert': ['dessert', 'cake', 'coffee'],
    'pricing': ['price','payment', 'cost', 'worth'],
    'parking': ['parking', 'car', 'location', 'space', 'place', 'spot', 'city'],
    'beverage': ['beverage', 'drink', 'beer', 'wine', 'bar', 'vodka'],
    }


def sentiment_text(text, concepts):

    #Word lemmatizer
    lemmatizer = WordNetLemmatizer()
    #Word Tokensizer
    sentences = nltk.sent_tokenize(text)

    #Stop words
    stop_words = set(stopwords.words('english'))

    #Customed Stop words for generalization
    new_stop_words = ['acme', 'nola', 'oyster', 'orleans']
    stop_words.update(new_stop_words)

    preprocessed_sentences = []
    for sent in sentences:
        #lower case
        words = [word.lower() for word in nltk.word_tokenize(sent) if word.isalpha()]
        filtered_words = [word for word in words if word not in stop_words]
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
        preprocessed_sentences.append(' '.join(lemmatized_words))  # joining words back into sentences

    concept_sentiments = {}
    for concept in concepts:
        concept = concept.lower()
        sentiment_scores = []

        for sentence in preprocessed_sentences:
            if any(word in sentence.split() for word in concepts[concept]):
                #use ABSA model to predict sentiment
                inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {concept} [SEP]", return_tensors="pt")
                outputs = absa_model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                probs = probs.detach().numpy()[0]

                #Selecting Max probablility for sentiment
                sentiment = ["negative", "neutral", "positive"][probs.argmax()]

                if sentiment == 'positive':
                    sentiment_scores.append(1)
                elif sentiment == 'negative':
                    sentiment_scores.append(-1)
                else:
                    sentiment_scores.append(0)

        if sentiment_scores:
            overall_sentiment = sum(sentiment_scores)
            if overall_sentiment > 0:
                concept_sentiments[concept] = 1
            else:
                # if overall sentiment is negative or neutral, we assign it as negative
                concept_sentiments[concept] = -1
        
        else:
          #No sentiment detected
          concept_sentiments[concept] = 0


    return concept_sentiments

def create_pos_neg_columns(data):
    # Create a new DataFrame
    new_df = pd.DataFrame()

    # Iterate over each keyword-value pair in the dictionary
    for keyword, value in data.items():
        # Create positive and negative column names
        pos_col = f'{keyword}_pos'
        neg_col = f'{keyword}_neg'

        # Assign values to the new DataFrame based on positive and negative conditions
        new_df[pos_col] = [1 if value > 0 else 0]
        new_df[neg_col] = [1 if value < 0 else 0]

    return new_df

df = pd.read_csv('cleaned_data.csv')

#Use only review text data for sentiment analysis
data_list = df['text'].reset_index(drop=True).values.tolist()

#Initial setting for sentiment analysis
x_example = sentiment_text(data_list[0], concepts)
quality = create_pos_neg_columns(x_example)

#Find sentiment for each review
for i in range(1,len(df)):
    sent_test = sentiment_text(data_list[i], concepts)  
    sent_pos_neg = create_pos_neg_columns(sent_test)
    quality = pd.concat([quality, sent_pos_neg])

quality = quality.reset_index(drop=True)

stars = df['stars'].tolist()
date = df['date'].tolist()
user_id = df['user_id'].tolist()

#Add stars, date, user_id to the quality dataframe
quality['stars'] = stars
quality['date'] = date
quality['user_id'] = user_id

#Save the dataframe to csv file
quality.to_csv('quality_dimension.csv', index=False)