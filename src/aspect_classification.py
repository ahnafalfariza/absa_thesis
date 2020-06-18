from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from clean_text import CleanText
import numpy as np
import pandas as pd


class AspectClassification:
    def __init__(self):
        self.data = pd.read_csv('../data/ac_dataset.csv', index_col=0)
        self.data['Target'] = self.convert_targets(self.data.drop(['Reviews'], axis=1))

        self.X = self.data['Reviews']
        self.y = self.data['Target']

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, train_size=0.8, random_state=0
        )

        self.pipeline = Pipeline([
            ('preprocess', CleanText()),
            ('vect', CountVectorizer(ngram_range=(1, 2))),
            ('clf', OneVsRestClassifier(
                LogisticRegression()))
        ])
        self.model = None

    def cross_validation(self):
        totalAcc = []
        totalPrec = []
        totalRec = []
        totalF1 = []

        confTot = 0

        kf = KFold(n_splits=10)
        i = 1
        for train_index, test_index in kf.split(self.X, self.y):
            model = self.train(self.X[train_index], self.y[train_index])
            conf, accuracy, precision, recall, f1 = self.evaluate(model, self.X[test_index], self.y[test_index])
            confTot += conf
            totalAcc.append(accuracy)
            totalPrec.append(precision)
            totalRec.append(recall)
            totalF1.append(f1)
            i += 1

        # print(confTot)
        print('acc', self.get_average(totalAcc))
        print('pre', self.get_average(totalPrec))
        print('rec', self.get_average(totalRec))
        print('f1', self.get_average(totalF1))

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

    def get_average(self, metrics):
        return np.average(np.array(metrics), axis=0)

    def train(self, input_value, target_value):
        mlb = MultiLabelBinarizer()
        target = mlb.fit_transform(target_value)

        model = self.pipeline.fit(input_value, target)
        return model

    def evaluate(self, model, input_value, target_value):
        target = MultiLabelBinarizer().fit_transform(target_value)
        prediction = model.predict(input_value)

        conf = confusion_matrix(target.argmax(axis=1), prediction.argmax(axis=1))
        acc = [accuracy_score(target[i], prediction[i]) for i in range(5)]
        pre = precision_score(target, prediction, average=None)
        rec = recall_score(target, prediction, average=None)
        f1 = f1_score(target, prediction, average=None)

        # print(classification_report(target, prediction, target_names=['#GENERAL', '#FEATURE', '#PRICE', '#CAMERA', '#DESIGN#SCREEN'], zero_division=1))

        return conf, acc, pre, rec, f1


if __name__ == "__main__":
    test = AspectClassification()
    test.cross_validation()
    # test.evaluate()
