import pandas as pd
import numpy as np
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
import nltk


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "postag": postag,
        "postag[:2]": postag[:2],
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update(
            {
                "-1:word.lower()": word1.lower(),
                "-1:word.istitle()": word1.istitle(),
                "-1:word.isupper()": word1.isupper(),
                "-1:postag": postag1,
                "-1:postag[:2]": postag1[:2],
            }
        )
    else:
        features["BOS"] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update(
            {
                "+1:word.lower()": word1.lower(),
                "+1:word.istitle()": word1.istitle(),
                "+1:word.isupper()": word1.isupper(),
                "+1:postag": postag1,
                "+1:postag[:2]": postag1[:2],
            }
        )
    else:
        features["EOS"] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


class AspectExtraction:
    def __init__(self):
        data = pd.read_csv("../data/ae_dataset_sample.csv", delimiter=";")
        data = data.dropna(subset=["Aspect"]).apply(lambda x: x.astype(str).str.lower())

        sentences = data.apply(
            lambda x: self.tag_word(x["Reviews"], x["Aspect"]), axis=1
        )

        self.X = np.array([sent2features(s) for s in sentences])
        self.y = np.array([sent2labels(s) for s in sentences])

        totalAcc = []
        totalPrec = []
        totalRec = []
        totalF1 = []
        totalconf = 0

        kf = KFold(n_splits=10)
        i = 1
        for train_index, test_index in kf.split(self.X, self.y):
            print('iteration', i)
            model = self.train(self.X[train_index], self.y[train_index])

            conf, accuracy, precision, recall, f1 = self.evaluate(model, self.X[test_index], self.y[test_index])
            totalconf += conf
            totalAcc.append(accuracy)
            totalPrec.append(precision)
            totalRec.append(recall)
            totalF1.append(f1)
            i += 1

        print(totalconf)
        print('acc', self.get_average(totalAcc))
        print('pre', self.get_average(totalPrec))
        print('rec', self.get_average(totalRec))
        print('f1', self.get_average(totalF1))

    def train(self, input_value, target_value):
        crf = sklearn_crfsuite.CRF()
        return crf.fit(input_value, target_value)

    def evaluate(self, model, input_value, target_value):
        prediction = model.predict(input_value)
        prediction = list(np.concatenate(prediction).flat)
        target_value = list(np.concatenate(target_value.tolist()).flat)

        conf = confusion_matrix(target_value, prediction)
        acc = accuracy_score(target_value, prediction)
        pre = precision_score(target_value, prediction, average=None)
        rec = recall_score(target_value, prediction, average=None)
        f1 = f1_score(target_value, prediction, average=None)

        return conf, acc, pre, rec, f1

    def print_transitions(self, trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

    def print_state_features(self, state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-8s %s" % (weight, label, attr))

    def get_average(self, metrics):
        return np.average(np.array(metrics), axis=0)

    def tag_word(self, word, term):
        term = term.split(", ")

        word = nltk.word_tokenize(word)
        word, tag = zip(*nltk.pos_tag(word))
        aspect = [None] * len(word)

        for i in range(len(word)):
            if word[i] in term or word[i] == term:
                aspect[i] = "ASPECT-B"
            elif "%s %s" % (word[i - 1], word[i]) in term:
                aspect[i - 1] = "ASPECT-B"
                aspect[i] = "ASPECT-I"
            else:
                aspect[i] = "OTHERS"

        return list(zip(word, tag, aspect))


if __name__ == "__main__":
    test = AspectExtraction()
