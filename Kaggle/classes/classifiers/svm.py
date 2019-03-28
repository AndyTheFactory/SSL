import collections
import json

import nltk
from nltk.corpus import wordnet as wn, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

from sklearn.preprocessing import LabelEncoder
#from sklearn import svm
from thundersvmScikit import *

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import pandas as pd
import joblib

class SVMClassifier(object):
    model_name = 'svm'

    def __init__(self, config):
        self.num_input_tokens = config.get('num_input_tokens')
        self.max_input_seq_length = config.get('max_input_seq_length')
        self.num_target_tokens = config.get('num_target_tokens')
        self.word2idx = config.get('word2idx')
        self.idx2word = config.get('idx2word')
        self.config = config

        self.model = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')



    def load_weights(self, weight_file_path):
        self.model=joblib.load(weight_file_path)

    @staticmethod
    def transform_input_text( texts):
        res=texts.to_frame('text')
        res['text'] = [entry.lower() for entry in res['text']]
        res['text'] = [nltk.word_tokenize(entry) for entry in res['text']]

        #texts=[a.lower() for a in texts]
        #texts=[nltk.word_tokenize(a) for a in texts]

        word_map= collections.defaultdict(lambda : wn.NOUN)
        word_map['J']=wn.ADJ
        word_map['V']=wn.VERB
        word_map['R']=wn.ADV

        res['text_final']=''

        for index, row in res.iterrows():
            # Declaring Empty List to store the words that follow the rules for this step
            Final_words = []
            # Initializing WordNetLemmatizer()
            word_Lemmatized = nltk.WordNetLemmatizer()
            # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
            for word, tag in nltk.pos_tag(row['text']):
                # Below condition is to check for Stop words and consider only alphabets
                if word not in stopwords.words('english') and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(word, word_map[tag[0]])
                    Final_words.append(word_Final)
            # The final processed set of words for each iteration will be stored in 'text_final'
            res.loc[index, 'text_final'] = str(Final_words)
        return res
        # for token in texts:
        #     x = []
        #     word_Lemmatized = nltk.WordNetLemmatizer()
        #
        #     for word, tag in nltk.pos_tag(token):
        #         if word not in stopwords.words('english') and word.isalpha():
        #             w=word_Lemmatized.lemmatize(word,word_map[tag[0]])
        #             x.append(w)
        #     temp.append(x)
        #print(temp.shape)
        #return temp


    def fit(self, Xtrain, Ytrain, Xtest=None, Ytest=None, epochs=None, file_prefix=None):

        model_dir_path = self.config.getPath('models')

        filename = self.model_name
        if file_prefix != None:
            filename = file_prefix+'_'+self.model_name

        filename = model_dir_path + '/' + filename
        print("start text transformation --->")
        encoder=LabelEncoder()
        Ytrain = encoder.fit_transform(Ytrain)
        Ytest = encoder.fit_transform(Ytest)

        Xtrain = self.transform_input_text(Xtrain)
        if Xtest is not None:
            Xtest = self.transform_input_text(Xtest)

        # save model weights
        config_file_path = filename + '-config.json'
        weight_file_path = filename + '-weights.h5'
        print("end text transformation <----")


        open(config_file_path, 'w').write(self.config.to_json())

        print("start text TFIDF--->")
        Tfidf_vect = TfidfVectorizer(max_features=5000)
        vocab=Xtrain.append(Xtrain)
        Tfidf_vect.fit(vocab['text_final'])
        print("end text TFIDF<----")

        #Xtrain=pd.Series(Xtrain)
        #Xtest=pd.Series(Xtest)

        Train_X_Tfidf = Tfidf_vect.transform(Xtrain['text_final'])
        Test_X_Tfidf = Tfidf_vect.transform(Xtest['text_final'])

        print("start text FIT--->")
        self.model.fit(Train_X_Tfidf, Ytrain)
        print("end text FIT<---")

        # save model architecture
        #architecture_file_path = filename + '-architecture.joblib'
        #joblib.dump(self.model,architecture_file_path)
        #open(architecture_file_path, 'w').write(pickle.dumps(SVM))
        self.model.save_to_file(filename + '-model.txt')

        predictions_SVM = self.model.predict(Test_X_Tfidf)
        print(self.model_name," Accuracy Score -> ", accuracy_score(predictions_SVM, Ytest) * 100)
        print(self.model_name," F1 Score -> ", f1_score(y_pred=predictions_SVM, y_true=Ytest) )
        print(self.model_name," Recall Score -> ", recall_score(y_pred=predictions_SVM, y_true=Ytest) )

        df = pd.read_csv(self.config.getPath('data') + "/test.csv")
        X = self.transform_input_text(df['question_text'])
        #Y = encoder.fit_transform(df['target'])

        print("start text Predictions--->")
        X_Tfidf = Tfidf_vect.transform(X['text_final'])
        predictions_SVM = self.model.predict(X_Tfidf)
        df['predictions_y']=predictions_SVM
        df.to_csv(self.config.getPath('data') + "/test-result.csv")
        #print(self.model_name,"Test DATA Accuracy Score -> ", accuracy_score(predictions_SVM, Ytest) * 100)
        #print(self.model_name,"Test DATA F1 Score -> ", f1_score(y_pred=predictions_SVM, y_true=Ytest) )
        #print(self.model_name,"Test DATA Recall Score -> ", recall_score(y_pred=predictions_SVM, y_true=Ytest) )

        # plot_decision_regions(X=Train_X_Tfidf.toarray(),
        #                       y=Ytrain,
        #                       clf=SVM,
        #                       legend=2)
        #
        # # Update plot object with X/Y axis labels and Figure Title
        # plt.title('SVM Decision Region Boundary', size=16)

        return None

    def predict(self, x):
        is_str = False
        if type(x) is str:
            is_str = True
            x = [x]

        Xtest = self.transform_input_text(x)

        preds = self.model.predict(Xtest)
        if is_str:
            preds = preds[0]
        return preds
