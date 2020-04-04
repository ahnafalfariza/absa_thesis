from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from clean_text import CleanText
import numpy as np
import pandas as pd


class SentimentAnalysis:
    def __init__(self):
        self.data = pd.read_csv('../data/sa_dataset.csv', index_col=0)
        self.data['Reviews'] = CleanText().fit_transform(self.data['Reviews'])
        self.X, self.y = self.classify_data(self.data)
        self.target_names = ['#GENERAL', '#FEATURE', '#PRICE', '#CAMERA', '#DESIGN#SCREEN']
        self.pipeline = Pipeline([
            # ('preprocess', CleanText()),
            ('vect', CountVectorizer(ngram_range=(1, 2))),
            ('clf', LogisticRegression())
        ])
        self.model = []

    def classify_data(self, data):
        X = [[], [], [], [], []]
        y = [[], [], [], [], []]
        for row in data.itertuples():
            for j in range(1, 6):
                if(row[j+1] != '-'):
                    X[j-1].append(row[1])
                    y[j-1].append(row[j+1])
        return X, y

    def split_train_test(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.75)
        return X_train, X_val, y_train, y_val

    def train(self):
        for i in range(5):
            X_train, X_val, y_train, y_val = self.split_train_test(self.X[i], self.y[i])
            self.model.append(self.pipeline.fit(X_train, y_train))
            prediction = self.model[i].predict(X_val)
            print('Target=%s' % self.target_names[i])
            print(' f1           = %s' % (f1_score(y_val, prediction, average=None)))
            print(' precision    = %s' % (precision_score(y_val, prediction, average=None)))
            print(' recall       = %s' % (recall_score(y_val, prediction, average=None)))
            # print(classification_report(y_val, prediction, zero_division=1))


if __name__ == "__main__":
    sa = SentimentAnalysis()
    sa.train()
