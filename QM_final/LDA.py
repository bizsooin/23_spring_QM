import os
import pandas as pd
import numpy as np
from pprint import pprint

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import spacy

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from  lang_detect import clearning
import matplotlib.pyplot as plt

#Importing the dataset
df = pd.read_csv('review_most.csv')

#Data cleaning
df_clean = cleaning(df)
df_clean = df.sort_values(by='date')
text_review = df_clean['text'].tolist()


#Customed Stop words for generalization
stop_words = set(stopwords.words('english'))
new_stop_words = ['acme', 'nola', 'oyster', 'orleans']
stop_words.update(new_stop_words)

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

#Used Bigrams language model
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

#Lemmatization
def lemmatization(texts, allowed_postags):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

all_words = list(sent_to_words(text_review))

bigram = gensim.models.Phrases(all_words, min_count=5, threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
all_words_nonstops = remove_stopwords(all_words)
all_words_bigrams = make_bigrams(all_words_nonstops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

#Using only noun for better LDA
all_word_lemmatized = lemmatization(all_words_bigrams, allowed_postags=['NOUN'])
original = all_word_lemmatized

id2word = corpora.Dictionary(original)
texts = original
corpus = [id2word.doc2bow(text) for text in texts]

#LDA model Example
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=8, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

pprint(lda_model.print_topics())

#Select best number of topic using Coherence score
lda_model_coherence = []
for i in range (2,14):
    print(i)
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=i, random_state=100,
                                           update_every=1, chunksize=100,
                                           passes=10, alpha='auto',
                                           per_word_topics=True)
    cm = CoherenceModel(model=lda_model, texts=all_word_lemmatized, dictionary=id2word, coherence='c_v')
    coherence = cm.get_coherence()
    lda_model_coherence.append(coherence)

plt.plot(range(2, 14),lda_model_coherence)
plt.xlabel('Number of topics')
plt.ylabel('Coherence score')
plt.title('How many topics ? (Closer to 0 = worse)')
plt.show()