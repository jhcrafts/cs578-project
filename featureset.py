from abc import ABCMeta, abstractmethod

class FeatureSet(object):
    """description of class"""

    @abstractmethod
    def extractfeatures(self,trainingexamples,testexamples):
        pass   
  
    @abstractmethod
    def formattedtrainingexamples(self):
        pass 
    
    @abstractmethod
    def formattedtestingexamples(self):
        pass 

    @abstractmethod
    def label(self,fmt_label):
        pass

    @abstractmethod
    def labelfromexample(self,fmt_example):
        pass

    @abstractmethod
    def formattedlabels(self):
        pass

    @abstractmethod
    def formattedexamplevectorlength(self):
        pass

    @abstractmethod
    def idfromexample(self,example):
        pass