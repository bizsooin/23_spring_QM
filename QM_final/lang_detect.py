from langdetect import detect
import numpy as np
import pandas as pd
import re
import os

def cleaning(data):
    df = data.copy()

    # Convert column names to lowercase
    df.columns = map(str.lower, df.columns)

    # Select 'text', 'stars', and 'date' columns
    df = df[['text', 'stars', 'date', 'user_id']]

    # Remove web_urls
    def remove_web_urls(text):
        return re.sub(r'https?://\S+', ' ', text)

    df['text'] = df['text'].apply(remove_web_urls)

    # Remove Tags
    def remove_tags(text):
        return re.sub(r'@\w*', ' ', text)

    df['text'] = df['text'].apply(remove_tags)


    # Remove punctuations
    def remove_apostrophe(text):
        return re.sub(r"'s\b", "", text)

    df['text'] = df['text'].apply(remove_apostrophe)

    # Remove special characters
    def remove_special_chars(text):
        return re.sub(r"[^a-zA-Z0-9\s]", ' ', text)

    df['text'] = df['text'].apply(remove_special_chars)

    # Remove numbers
    def remove_number(text):
        return re.sub(r'[\d]', ' ', text)

    df['text'] = df['text'].apply(remove_number)

    # Remove extra spaces
    def remove_with(text):
        return re.sub(r'\bwith\b', ' ', text)

    df['text'] = df['text'].apply(remove_with)

    # Only use English reviews
    def is_english(text):
        try:
            lang = detect(text)
            return lang == 'en'
        except:
            return False

    df = df[df['text'].apply(is_english)]

    return df