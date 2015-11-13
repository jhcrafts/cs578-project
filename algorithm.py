from abc import ABCMeta, abstractmethod

class Algorithm(object):
    "Abstract Base Class for Algorithm Classes"
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def extractfeatures(self,trainingexamples):
        pass   
    
    @abstractmethod
    def formatexample(self,example):
        pass 

    @abstractmethod
    def train(self,trainingexamples):
        pass

    @abstractmethod
    def predict(self,example):
        pass

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def description(self):
        pass

    @abstractmethod
    def label(self,fmt_label):
        pass

    @abstractmethod
    def labelfromexample(self,fmt_example):
        pass



