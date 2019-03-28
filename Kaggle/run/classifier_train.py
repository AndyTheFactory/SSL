from __future__ import print_function

import datetime

import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from classes.utils.plot_utils import plot_and_save_history
from classes.classifiers.lstm import LstmClassifier
from classes.classifiers.bilstm import BiLstmClassifier
from config_class import Config
from classes.factory import ClassifierFactory
import numpy as np
from classes.embeddings.factory import EmbeddingsFactory

np.random.seed(42)

TRAINING_DATA='train_equil.csv'
config=None

def load_config(**kwargs):
    print('loading csv file ...')
    global config

    df = pd.read_csv(Config.getPath('data') + '/' + TRAINING_DATA)
    df=df.sample(50000)
    X = df['question_text']
    Y = df['target']

    print('preparing configuration...')

    config = ClassifierFactory.getConfig(X,Y,json_file=None,**kwargs)

    #Two classes - Fake=0, Reliable=1
    config.set('num_target_tokens',2)

def train_vanilla(classifier):
    global config
    print('configuration extracted from input texts ...')

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(config.getData(), config.getLabels(), test_size=0.2, random_state=42)

    print('training size: ', len(Xtrain))
    print('testing size: ', len(Xtest))

    print('start fitting ...')
    history = classifier.fit(Xtrain, Ytrain, Xtest, Ytest,epochs=config.get('epochs'))
    if (history!=None):
        history_plot_file_path = Config.getPath('reports') + '/' + classifier.model_name + '-history.png'
        plot_and_save_history(history, classifier.model_name, history_plot_file_path)


def train_experiment(classifier):
    print('loading csv file ...')
    global config

    df = pd.read_csv(Config.getPath('data') + '/' + TRAINING_DATA)

    df=df.sample(20000)

    X = df['question_text']
    Y = df['target']

    print('splitting data...')

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    print('training size: ', len(Xtrain))
    print('testing size: ', len(Xtest))

    print('start fitting ...')

    # max_sequence, vocab_size,  lstm_units, dropout
    experiment = [
        [20, 5000, 64, 0.2],      #0
        [35, 5000, 64, 0.2],      #1
        [50, 5000, 64, 0.2],      #2
        [100,5000, 64, 0.2],      #3

        [50, 5000, 128, 0.2],  # 4
        [50, 5000, 256, 0.2],  # 5
        [50, 5000, 512, 0.2],  # 6

        [50, 2000, 64, 0.2],  # 7
        [50, 3000, 64, 0.2],  # 8
        [50, 4000, 64, 0.2],  # 9
        [50, 5000, 64, 0.2],  # 10
        [50, 6000, 64, 0.2],  # 11
        [50, 7000, 64, 0.2],  # 12
        [50, 8000, 64, 0.2],  # 13
        [50, 9000, 64, 0.2],  # 14

        [50, 5000, 64, 0.1],  # 15
        [50, 5000, 64, 0.2],  # 16
        [50, 5000, 64, 0.3],  # 17
        [50, 5000, 64, 0.4],  # 18
    ]
    i = 0
    for max_seq, vocab_siz,  lstm_u, drop in experiment:

        config = Config(X,Y,max_seq,vocab_siz,2,'glove')

        print("%s starting experiment ... %d" % (datetime.datetime.now(),i))
        #model=ClassifierFactory.getLSTM(**{'config':config,'lstm_units':lstm_u,'dropout':drop})
        model=ClassifierFactory.getLSTM(**{'config':config,'lstm_units':lstm_u,'dropout':drop})


        history = model.fit(Xtrain, Ytrain, Xtest, Ytest,epochs=10,file_prefix='experiment-%i' % i)

        history_plot_file_path = Config.getPath('reports') + '/' + model.model_name + (
                    '_experiment_%d' % i) + '-history.png'
        plot_and_save_history(history, model.model_name, history_plot_file_path)
        i += 1

def main_svm():
    load_config()
    c=ClassifierFactory.getSVM()
    train_vanilla(c)

def main():
    load_config(**{'embedding':'glove','max_input_seq_length':100,'max_vocab_size':8000,'epochs':20})
    c=ClassifierFactory.getLSTM(**{'lstm_units':64})
    #c=ClassifierFactory.getLSTM(**{'lstm_units':64,'dropout':0.2,'epochs':15})
    train_vanilla(c)
def main_experiment():
    train_experiment('lstm')
    #load_config(**{'embedding':'glove','max_input_seq_length':50,'max_vocab_size':3000})
    #c=ClassifierFactory.getLSTM(**{'lstm_units':64,'dropout':0.2,'epochs':15})
    #c=ClassifierFactory.getBiLSTM(**{'lstm_units':64})
    #train_vanilla(c)
if __name__ == '__main__':
    main()
