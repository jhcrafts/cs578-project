import algorithm
import math

class GradientDescentBinaryClassifier:
    "Gradient Descent Binary Classifier"
    mytargetlabel = 0
    weights = None
    bias = 0.0
    name = None
    lmbd = 0.01
    stepsize = 0.01   
    
    def __init__(self, targetlabel, weightvectorsize, name):
        self.mytargetlabel = targetlabel
        self.weights = [0.0]*weightvectorsize 
        self.bias = 0.0
        self.name = name
   
    def trainclassifier(self, examplevectors, maxiterations):        
        regularizer = GradientDescentBinaryClassifier.l1reg #GradientDescentBinaryClassifier.l2reg                        
        trainingaccuracy = list()
        nonzerolossexamples = list()
        for i in range(maxiterations):
            g = [0.0]*len(self.weights);
            gb = 0.0
            n = 0.0
            total = 0.0
            correct = 0.0
            for example in examplevectors:
                sign = 1 if example[0] == self.mytargetlabel else -1
                dotproduct = 0.0
                for index in example[1]:
                    dotproduct += self.weights[index]                
                if (sign*(dotproduct + self.bias)) <= 1.0:
                    n += 1
                    for index in example[1]:                
                        g[index] += sign
                    gb += sign    
                #Calculate Training Accuracy Inline - This saves computation effort          
                if (sign*(dotproduct + self.bias)) > 0.0:
                    correct += 1  
                total += 1                            
            trainingaccuracy.append(correct/total)
            nonzerolossexamples.append(n/total)
            for k in range(len(g)):
                g[k] = g[k] - regularizer(self,self.weights[k], GradientDescentBinaryClassifier.lmbd)
                self.weights[k] = self.weights[k] + GradientDescentBinaryClassifier.stepsize*g[k]
            self.bias = self.bias + GradientDescentBinaryClassifier.stepsize*gb        
        print(str(self.mytargetlabel) + "," + str(trainingaccuracy) + "," + str(nonzerolossexamples))
        magnitude = 0.0
        for i in range(len(self.weights)):
            magnitude += math.pow(self.weights[i], 2)
        magnitude += math.pow(self.bias, 2)
        if (magnitude != 0.0):
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i]/magnitude
            self.bias = self.bias/magnitude         
        return    
    
    def l1reg(self,weight,lmbd):
        if (weight > 0.0):
            return lmbd
        elif (weight < 0.0):
            return -1.0*lmbd
        else:
            return 0

    def l2reg(self,weight,lmbd):
        return weight*lmbd 

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

class AlgGradientDescentOVA(algorithm.Algorithm):
    "Implementation of Gradient Descent One vs. All Classifier with base class 'algorithm'"
    binaryclassifiers = list()         

    def train(self, trainingexamples, examplevectorlength, labels, iterations):        
        AlgGradientDescentOVA.binaryclassifiers = list()
        ## create a classifier for each label
        for cuisine in labels:
            AlgGradientDescentOVA.binaryclassifiers.append(GradientDescentBinaryClassifier(cuisine, examplevectorlength,""))
        ## train each classifier on every example
        for classifier in AlgGradientDescentOVA.binaryclassifiers:
            classifier.trainclassifier(trainingexamples, iterations)        

    def predict(self, example):
        results = list()        
        for classifier in AlgGradientDescentOVA.binaryclassifiers:
            result = classifier.predict(example)
            results.append([result,classifier.mytargetlabel])
        sortedresults = sorted(results, cmp=resultcompare)
        return sortedresults[0][1]

    def name(self):
        return "Gradient Descent One vs. All"

    def description(self):
        pass

def resultcompare(x,y):
    return -1 if (x[0] - y[0]) > 0.0 else 1 
