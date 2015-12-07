from abc import ABCMeta, abstractmethod

class Algorithm(object):
    "Abstract Base Class for Algorithm Classes"
    __metaclass__ = ABCMeta   

    @abstractmethod
    def train(self,trainingexamples, examplevectorlength, labels, iterations):
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



