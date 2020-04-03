from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd


class aspect_classification:
    def __init__(self):
        self.data = pd.read_csv('../data/ac_dataset.csv', index_col=0)
        self.X = self.data['Reviews']
        self.y = self.convert_targets(self.data.drop(['Reviews'], axis=1))

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, train_size=0.75
        )

        self.pipeline = Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 2))),
            ('clf', OneVsRestClassifier(
                LogisticRegression()))
        ])
        self.model = None

    def convert_targets(self, y):
        target_names = ['#GENERAL', '#FEATURE', '#PRICE', '#CAMERA', '#DESIGN#SCREEN']
        target_convert = []
        for row in np.array(y):
            target = []
            for i in range(len(target_names)):
                if row[i] == True:
                    target.append(target_names[i])
            target_convert.append(target)
        return target_convert

    def train(self):
        mlb = MultiLabelBinarizer()
        target = mlb.fit_transform(self.y_train)

        self.model = self.pipeline.fit(self.X_train, target)
        self.y_val = MultiLabelBinarizer().fit_transform(self.y_val)

    def evaluate(self):
        print(' f1           = %s' % (f1_score(self.y_val, self.model.predict(self.X_val), average=None)))
        print(' precision    = %s' % (precision_score(self.y_val, self.model.predict(self.X_val), average=None)))
        print(' recall       = %s' % (recall_score(self.y_val, self.model.predict(self.X_val), average=None)))

    def testPrint(self):
        print()


if __name__ == "__main__":
    test = aspect_classification()
    test.train()
    test.evaluate()
