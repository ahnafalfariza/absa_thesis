from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold
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
            ('vect', CountVectorizer(ngram_range=(1, 1))),
            ('clf', LogisticRegression())
        ])
        self.model = []

        self.a = []
        self.p = []
        self.r = []
        self.f = []

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
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=0)
        return X_train, X_val, y_train, y_val

    def evaluate(self, model, input_value, target):
        prediction = model.predict(input_value)

        acc = accuracy_score(target, prediction)
        pre = precision_score(target, prediction, average=None, zero_division=1)
        rec = recall_score(target, prediction, average=None, zero_division=1)
        f1 = f1_score(target, prediction, average=None, zero_division=1)

        cr = classification_report(target, prediction,  zero_division=1)

        return acc, pre, rec, f1, cr

    def cross_validation(self):
        kf = KFold(n_splits=10)

        for i in range(5):
            self.X[i] = np.array(self.X[i])
            self.y[i] = np.array(self.y[i])

            j = 1
            totalAcc = []
            totalPrec = []
            totalRec = []
            totalF1 = []

            for train_index, test_index in kf.split(self.X[i], self.y[i]):
                # print('iteration', j)
                model = self.train(self.X[i][train_index], self.y[i][train_index])
                accuracy, precision, recall, f1, cr = self.evaluate(model, self.X[i][test_index], self.y[i][test_index])
                # print(cr)
                totalAcc.append(accuracy)
                totalPrec.append(precision)
                totalRec.append(recall)
                totalF1.append(f1)
                j += 1
                # X_train, X_val, y_train, y_val = self.split_train_test(self.X[i], self.y[i])
                # self.model.append(self.pipeline.fit(X_train, y_train))
                # prediction = self.model[i].predict(X_val)
                # print('\nTarget=%s' % self.target_names[i])
                # print(pd.DataFrame({'review': X_val, 'target': y_val, 'prediction': prediction}))
                # print(' f1           = %s' % (f1_score(y_val, prediction, average=None)))
                # print(' precision    = %s' % (precision_score(y_val, prediction, average=None)))
                # print(' recall       = %s' % (recall_score(y_val, prediction, average=None)))
                # print(classification_report(y_val, prediction, zero_division=1))

            self.a.append(totalAcc)
            self.p.append(totalPrec)
            self.r.append(totalRec)
            self.f.append(totalF1)
            # print('acc', self.target_names[i], np.array(totalAcc))
            # print('pre', self.target_names[i], np.array(totalPrec))
            # print('rec', self.target_names[i], np.array(totalRec))
            # print('f1', self.target_names[i], np.array(totalF1))

        print('a', totalAcc)
        print('p', totalPrec)
        print('r', totalRec)
        print('f', totalF1)

    def train(self, input_value, target_value):
        model = self.pipeline.fit(input_value, target_value)
        return model

    def get_average(self, metrics):
        # return np.average(np.array(metrics), axis=0)
        pos, neg = 0, 0
        for a in metrics:
            # print(len(a))
            if(type(a) == list):
                neg += a[0]
                pos += a[1]
            else:
                accuracy = 5

        # print('pos', pos)
        # print('neg', neg)
        return [neg/len(metrics), pos/len(metrics)]


if __name__ == "__main__":
    sa = SentimentAnalysis()
    sa.cross_validation()
