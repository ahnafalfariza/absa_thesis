import pandas as pd
import numpy as np
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import nltk


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


class AspectExtraction:
    def __init__(self):
        data = pd.read_csv('../data/ae_dataset_sample.csv', delimiter=';')
        data = data.dropna(subset=['Aspect']).apply(lambda x: x.astype(str).str.lower())

        sentences = data.apply(lambda x: self.tag_word(x['Reviews'], x['Aspect']), axis=1)

        X = [sent2features(s) for s in sentences]
        y = [sent2labels(s) for s in sentences]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(X_train, y_train)

        y_pred = crf.predict(X_test)
        print(metrics.flat_classification_report(y_test, y_pred))

    def tag_word(self, word, term):
        term = term.split(', ')

        word = nltk.word_tokenize(word)
        word, tag = zip(*nltk.pos_tag(word))
        aspect = [None]*len(word)

        for i in range(len(word)):
            if word[i] in term or word[i] == term:
                aspect[i] = ('ASPECT-B')
            elif '%s %s' % (word[i-1], word[i]) in term:
                aspect[i-1] = ('ASPECT-B')
                aspect[i] = ('ASPECT-I')
            else:
                aspect[i] = ('OTHERS')

        return list(zip(word, tag, aspect))


if __name__ == "__main__":
    test = AspectExtraction()
