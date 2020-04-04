from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk


class CleanText(BaseEstimator, TransformerMixin):
    def remove_mentions(self, input_text):
        return re.sub(r'@\w+', '', str(input_text))

    def remove_urls(self, input_text):
        return re.sub(r'http.?://[^\s]+[\s]?', '', str(input_text))

    def emoji_oneword(self, input_text):
        return input_text.replace('_', '')

    def remove_punctuation(self, input_text):
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')
        return input_text.translate(trantab)

    def remove_digits(self, input_text):
        return re.sub(' \d+', '', str(input_text))

    def to_lower(self, input_text):
        return input_text.lower()

    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        whitelist = ["n't", "not", "no"]
        words = input_text.split()
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1]
        return " ".join(clean_words)

    def stemming(self, input_text):
        porter = PorterStemmer()
        words = input_text.split()
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)

    def get_word_counter(self, arr):
        cv = CountVectorizer()
        bow = cv.fit_transform(arr)
        word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
        word_counter = collections.Counter(word_freq)
        word_counter_df = pd.DataFrame(word_counter.most_common(20), columns=['word', 'freq'])
        print(word_counter_df)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
        plt.show()

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(
            self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.stemming)
        return clean_X


if __name__ == "__main__":
    data = pd.read_csv('../data/ac_dataset.csv', index_col=0)
    df = data['Reviews']
    pp = CleanText()
    preprocessed = pp.fit_transform(data['Reviews'])
    pp.get_word_counter(preprocessed)
