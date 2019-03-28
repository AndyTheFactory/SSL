from classes.classifiers.lstm import LstmClassifier
from classes.classifiers.bilstm import BiLstmClassifier
from classes.classifiers.svm import SVMClassifier
from config_class import Config
import inspect

class ClassifierFactory(object):
    __config=None
    @staticmethod
    def getLSTM(**kwargs):
        """
        Get LSTM Class with initialisation

        :param kwargs:
            config (Config): configuration class
            lstm_units (int): number of hidden layers
            dropout (float): dropout rate
        :return:
            LstmClassifier class:
        """
        if ('config' in kwargs):
            config=kwargs['config']
        else:
            config=ClassifierFactory.__config
        myargs={'config':config,
                'lstm_units':512}
        available_args=inspect.getargspec(LstmClassifier.__init__)
        for key, value in kwargs.items():
            if key in available_args:
                myargs[key]=value

        return LstmClassifier(**myargs)

    @staticmethod
    def getBiLSTM(**kwargs):
        if ('config' in kwargs):
            config=kwargs['config']
        else:
            config=ClassifierFactory.__config
        myargs={'config':config,
                'lstm_units':512}
        available_args=inspect.getargspec(LstmClassifier.__init__)
        for key, value in kwargs.items():
            if key in available_args:
                myargs[key]=value

        return BiLstmClassifier(**myargs)

    @staticmethod
    def getConfig(X=None,Y=None,json_file=None,joblib_file=None,**kwargs):
        """

        :param X: Text Data to be classified
        :param Y: True labels for training
        :param json_file: File to load Config
        :param kwargs:
            rest of config params:
                 max_input_seq_length
                 max_vocab_size
                 num_target_tokens
                 embedding
                 embedding_size
        :return:
        """
        if (ClassifierFactory.__config==None and joblib_file==None and json_file==None):
                ClassifierFactory.__config=Config(Txt_data=X,Txt_labels=Y,**kwargs)
        else:
            if (joblib_file!=None):
                ClassifierFactory.__config = Config.from_joblib(joblib_file)
            else:
                ClassifierFactory.__config=Config.from_json(json_file)

        return ClassifierFactory.__config

    @staticmethod
    def getClassifier(model_name,config=None):
        if model_name==LstmClassifier.model_name:
            return ClassifierFactory.getLSTM(config)
        if model_name==BiLstmClassifier.model_name:
            return ClassifierFactory.getBiLSTM(config)

    @staticmethod
    def getSVM(config=None):
        if (config==None):
            config=ClassifierFactory.__config
        return SVMClassifier(config)
