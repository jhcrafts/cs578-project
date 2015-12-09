import algorithm
import nltk
import math

class PerceptronBinaryClassifier:
    "Perceptron Binary Classifier"
    mytargetlabel = 0
    weights = None
    bias = 0.0
    cachedweights = None
    cachedbias = 0.0
    c = 1.0
    name = None    
    
    def __init__(self, targetlabel, weightvectorsize, name):
        self.mytargetlabel = targetlabel
        self.weights = [0.0]*weightvectorsize 
        self.cachedweights = [0.0]*weightvectorsize
        self.cachedbias = 0.0
        self.c = 1.0
        self.bias = 0.0
        self.name = name
    
    def trainclassifier(self, examplevectors, maxiterations):            
        iterations = 0
        for i in range(maxiterations):      
            iterations += 1
            exampleswithoutmistake = 0
            for example in examplevectors:
                sign = 1 if example[0] == self.mytargetlabel else -1
                dotproduct = 0.0
                for index in example[1]:
                    dotproduct += self.weights[index]
                if (sign*(dotproduct + self.bias) <= 0):
                    # mistake!
                    exampleswithoutmistake = 0
                    for index in example[1]:
                        self.weights[index] += sign
                        self.cachedweights[index] += (sign*self.c)
                    self.bias += sign
                    self.cachedbias += self.c*sign
                else:
                    exampleswithoutmistake += 1
                self.c += 1.0
            if (exampleswithoutmistake >= len(examplevectors)):
                break         
        magnitude = 0.0
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - self.cachedweights[i]/self.c
            magnitude += math.pow(self.weights[i], 2)
        self.bias = self.bias - (self.cachedbias/self.c) 
        magnitude += math.pow(self.bias, 2)
        if (magnitude != 0.0):
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i]/magnitude
            self.bias = self.bias/magnitude                   

    def predict(self, example):        
        dotproduct = 0.0
        for index in example[1]:
            dotproduct += self.weights[index]
        dotproduct += self.bias
        return dotproduct

    def name(self):
        return self.name

    def mytargetlabel(self):
        return mytargetlabel

import jc_features as fc
class AlgPerceptronOVA(algorithm.Algorithm):
    "Implementation of Perceptron One vs. All Classifier with base class 'algorithm'"
    binaryclassifiers = list()  
          
    def train(self, trainingexamples, examplevectorlength, labels, iterations):        
        AlgPerceptronOVA.binaryclassifiers = list()
        ## create a classifier for each label
        for cuisine in labels:
            AlgPerceptronOVA.binaryclassifiers.append(PerceptronBinaryClassifier(cuisine, examplevectorlength,""))
        ## train each classifier on every example
        for classifier in AlgPerceptronOVA.binaryclassifiers:
            classifier.trainclassifier(trainingexamples,iterations)        

    def predict(self, example):
        results = list()        
        for classifier in AlgPerceptronOVA.binaryclassifiers:
            result = classifier.predict(example)
            results.append([result,classifier.mytargetlabel])
        sortedresults = sorted(results, cmp=resultcompare)
        return sortedresults[0][1]

    def name(self):
        return "Perceptron One vs. All"
            
    def description(self):
        pass 

def resultcompare(x,y):
    return -1 if (x[0] - y[0]) > 0.0 else 1 
