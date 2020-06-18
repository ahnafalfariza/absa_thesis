from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import collections
import re
import string
import nltk


class CleanText(BaseEstimator, TransformerMixin):
    def remove_mentions(self, input_text):
        return re.sub(r"@\w+", "", str(input_text))

    def remove_urls(self, input_text):
        return re.sub(r"http.?://[^\s]+[\s]?", "", str(input_text))

    def emoji_oneword(self, input_text):
        return input_text.replace("_", "")

    def remove_punctuation(self, input_text):
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct) * " ")
        return input_text.translate(trantab)

    def remove_digits(self, input_text):
        return re.sub(" \d+", "", str(input_text))

    def to_lower(self, input_text):
        return input_text.lower()

    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words("english")
        whitelist = ["n't", "not", "no"]
        words = input_text.split()
        clean_words = [
            word
            for word in words
            if (word not in stopwords_list or word in whitelist) and len(word) > 1
        ]
        return " ".join(clean_words)

    def stemming(self, input_text):
        porter = PorterStemmer()
        words = input_text.split()
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)

    def lemmatization(self, input_text):
        lemmatizer = WordNetLemmatizer()
        words = input_text.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized_words)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        clean_X = (
            X.apply(self.remove_mentions)
            .apply(self.remove_urls)
            .apply(self.emoji_oneword)
            .apply(self.remove_punctuation)
            .apply(self.remove_digits)
            .apply(self.to_lower)
            .apply(self.lemmatization)
            .apply(self.remove_stopwords)
            .apply(self.stemming)
        )
        return clean_X


if __name__ == "__main__":
    data = pd.read_csv("../data/ac_dataset.csv", index_col=0)
    pp = CleanText()
    # preprocessed = pp.fit_transform(data['Reviews'])
    # for i in range(len(preprocessed)):
    #     print(i, preprocessed[i])
    #     print()
    # pp.get_word_counter(preprocessed)

    df = "Am i the only one who actually sees the bezels? They are Huge af!!! and no fingerprint under screen? yeah sure i give AROUND 1150euro for this! 10x apple... sucked even more!"
    caseFolding = pp.to_lower(df)
    removeUnusedWord = pp.remove_punctuation(
        pp.emoji_oneword(
            pp.remove_urls(pp.remove_mentions(pp.remove_digits(caseFolding)))
        )
    )
    lemmatization = pp.lemmatization(removeUnusedWord)
    removeStopWord = pp.remove_stopwords(lemmatization)
    stemming = pp.stemming(removeStopWord)
    print(caseFolding)
    print(removeUnusedWord)
    print(lemmatization)
    print(removeStopWord)
    print(stemming)
