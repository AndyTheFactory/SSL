from __future__ import print_function

from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split

from classes.factory import ClassifierFactory
from classes.utils.plot_utils import plot_confusion_matrix
import numpy as np

from config_class import Config

TESTING_DATA='train.csv'
config=None
np.random.seed(42)
TRAINING_DATA='train_equil.csv'


def load_config(model_name):
    print('loading csv file ...')
    global config

    config_file_path = Config.getPath('models') + '/' + model_name + '-config.joblib'

    config = ClassifierFactory.getConfig(joblib_file=config_file_path)

    #Two classes - Fake=0, Reliable=1
    config.set('num_target_tokens',2)

def main():
    global config
    USE_CLASSIFIER='lstm'

    weight_file_path = Config.getPath('models') + '/' + USE_CLASSIFIER + '-weights.h5'

    load_config(USE_CLASSIFIER)

    classifier=ClassifierFactory.getLSTM(**{'config':config})
    #Classifier(model_name=USE_CLASSIFIER,config=config)

    classifier.load_weights(weight_file_path)

    df = pd.read_csv(Config.getPath('data') + '/' + TESTING_DATA)

    Xtest = df['question_text']
    Ytest = df['target']

    print('extract configuration from input texts ...')

    print('testing size: ', len(Xtest))

    print('start predicting ...')
    pred = classifier.predict(Xtest)
    print(pred)
    score = metrics.accuracy_score(Ytest, pred)
    print("accuracy:   %0.3f" % score)
    cm = metrics.confusion_matrix(Ytest, pred, labels=[0, 1])
    plot_confusion_matrix(cm, classes=[0, 1])

def main_():
    predict_svm()

def predict_svm():
    global config
    load_config('svm')

    print('loading data...')
    df = pd.read_csv(Config.getPath('data') + '/' + TRAINING_DATA)

    X = df['question_text']
    Y = df['target']

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
    # Two classes - Fake=0, Reliable=1
    config.set('num_target_tokens', 2)

    classifier=ClassifierFactory.getSVM()

    print('training size: ', len(Xtrain))
    print('testing size: ', len(Xtest))


    print('start fitting ...')

    classifier.fit(Xtrain, Ytrain, Xtest, Ytest)


    df = pd.read_csv(Config.getPath('data') + '/' + TESTING_DATA)
    X = df['question_text']
    Y = df['target']

    pred=classifier.predict(X)

    score = metrics.accuracy_score(Ytest, pred)
    f1score = metrics.f1_score(Ytest, pred)
    print("accuracy:   %0.3f" % score)
    print("f1 score:   %0.3f" % f1score)

    cm = metrics.confusion_matrix(Ytest, pred, labels=[0, 1])
    plot_confusion_matrix(cm, classes=[0, 1])


if __name__ == '__main__':
    main()