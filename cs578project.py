# CS578 Project
# Francois Sanson
# Jordan Crafts


import sys
from math import *
import copy
import random
import json
import numpy as np
import csv
import os

### main
### ----
### The main program loop
### You should modify this function to run your experiments
##
def parseArgs(args):
  """Parses arguments vector, looking for switches of the form -key {optional value}.
  For example:
	parseArgs([ 'template.py', '-a', 1, '-i', 10, '-f', 1 ]) = {'-t':1, '-i':10, '-f':1 }"""
  args_map = {}
  curkey = None
  for i in xrange(1, len(args)):
    if args[i][0] == '-':
      args_map[args[i]] = True
      curkey = args[i]
    else:
      assert curkey
      args_map[curkey] = args[i]
      curkey = None
  return args_map

def validateInput(args):
    args_map = parseArgs(args)

    algorithm = 1 # 1: Perceptron OVA, 2: Decision Tree    
    forcefeatureextraction = False
    printtrainingdatastats = False    

    if '-a' in args_map:
      algorithm = int(args_map['-a'])    
    if '-f' in args_map:
      forcefeatureextraction = True
    if '-d' in args_map:
      printtrainingdatastats = True

    assert algorithm in [1, 2]

    return [algorithm, forcefeatureextraction, printtrainingdatastats]

##### Begin JHC Code
##predict a single example
def predict_one(weights, example,threshold):
    dotproduct = 0.0
    for i in range(1,len(example),1):
        dotproduct += weights[example[i]]
    return -1 if dotproduct <= threshold else 1

def predictperceptronprimal(weights,bias,example):
    return predict_one(weights,example,0.0-bias)

def predictwinnow(weights,bias,example):
    return predict_one(weights,example,len(weights))

def predictperceptronaverage(weights,bias,example):
    return predict_one(weights,example,0.0-bias)

##Perceptron
#-----------
def perceptronprimal(maxIterations, ngrams, trainingdata):
    weights = [0.0]*len(ngrams)    
    bias = 0
    iterations = 0.0
    for i in range(maxIterations):      
        iterations += 1
        exampleswithoutmistake = 0
        for example in trainingdata:
            sign = int(example[0]);
            dotproduct = 0
            for j in range(1,len(example),1):
                dotproduct += weights[example[j]]
            if (sign*(dotproduct + bias) <= 0):
                # mistake!
                exampleswithoutmistake = 0
                for k in range(1,len(example),1):
                    weights[example[k]] += sign
                bias += sign
            else:
                exampleswithoutmistake += 1
        if (exampleswithoutmistake >= len(trainingdata)):
            break
    return weights, bias, iterations

##Perceptron Dual
#-------
def perceptronaverage(maxIterations, ngrams, trainingdata):
    weights = [0.0]*len(ngrams)
    bias = 0.0
    cachedweights = [0.0]*len(ngrams)
    cachedbias = 0.0
    c = 1.0    
    iterations = 0.0
    for i in range(maxIterations):      
        iterations += 1
        exampleswithoutmistake = 0
        for example in trainingdata:
            sign = int(example[0]);
            dotproduct = 0
            for j in range(1,len(example),1):
                dotproduct += weights[example[j]]
            if (sign*(dotproduct + bias) <= 0):
                # mistake!
                exampleswithoutmistake = 0
                for k in range(1,len(example),1):
                    weights[example[k]] += sign
                    cachedweights[example[k]] += (sign*c)
                bias += sign
                cachedbias += c*sign
            else:
                exampleswithoutmistake += 1
            c += 1.0
        if (exampleswithoutmistake >= len(trainingdata)):
            break 
    retweights = copy.deepcopy(cachedweights)
    for i in range(len(retweights)):
        retweights[i] = weights[i] - retweights[i]/c
    retbias = bias - (cachedbias/c)
    return retweights,retbias,iterations

def runperceptronOVA(trainingexamples):
    trainingaccuracy = 0.0
    return trainingaccuracy
##### End JHC Code

##### Begin FS Code
def rundecisiontree(trainingexamples):
    trainingaccuracy = 0.0
    return trainingaccuracy
##### End Fs Code
##
##class Example:
##    'Class to encapsulate data examples'
##    featureValues = dict()
##
##    def __init__(self, label):
##        self.label = label
##        self.featureValues = dict()
##
##    def label(self):
##        return self.label
##
##    def addfeature(self, feature,value):
##        self.featureValues[feature] = value
##
##    def remfeature(self, feature):
##        del self.featureValues[feature]
##
##    def value(self, feature):
##        if (feature in self.featureValues.keys()):
##            return self.featureValues[feature]
##        else:
##            return None
##
##    def features(self):
##        return self.featureValues
##
#%%
def entropy(featurefreq,featuresubset,cuisinedict):

    totalfeatures=set()
    for subset in featuresubset: # gather all the feature in one set
        totalfeatures=totalfeatures.union(set(subset))
    totalfeatures=listodict(list(totalfeatures)) # turn it into a dict for indexing
    entropy=np.zeros(len(totalfeatures))
    for ingredient in totalfeatures:
        totalfreq=0.
        for cuis in cuisinedict:
            i=cuisinedict[cuis]
            try :
                f=featurefreq[i][featuresubset[i][ingredient]]
            except KeyError:
                f=0.
            totalfreq+=f
        for cuis in cuisinedict:
            i=cuisinedict[cuis]
            try :
                p=featurefreq[cuisinedict[i]][featuresubset[i][ingredient]]/totalfreq
                entropy[totalfeatures[ingredient]]+= -p*log(p)
            except KeyError:
                pass
    return(entropy,totalfeatures)
#%%
def completestat(examples):
    """
    Based on the examples computes the frequency of every ingredient for
    for every type of cuisine it can be useful to compute entropy
    the values are stored in numpy arrays
    -featurefreq[cuisine][ingredient] = freq of that ingredient
    indexed by
    -dictionnary cuisinedict (type of cuisine :chinese mexican ...) returns
    cuisine index
    -dictionnaries featuresubset[i]=  the dictionnary of ingredients for
    cuisine i  retunrs ingredient index
    """
    featuresubset=np.empty(20,dtype=object)
    for i in range(len(featuresubset)):
        featuresubset[i]=set()
    cuisinedict={}
    labels = set()
    for example in examples:
        if not(example["cuisine"] in labels): # to build the dictionnary index on the fly
            labels.add(example["cuisine"])
            cuisinedict[example["cuisine"]]=len(labels)-1
        for ingredient in example["ingredients"]:
            featuresubset[cuisinedict[example["cuisine"]]].add(ingredient)
    featurefreq=np.empty(20,dtype=object)
    for i,subset in enumerate(featuresubset):
        featurefreq[i]=np.zeros(len(subset))
        featuresubset[i]=listodict(list(subset))
    for example in examples:
        cuisine=cuisinedict[example["cuisine"]]
        for ingredient in example["ingredients"]:
            featurefreq[cuisine][featuresubset[cuisine][ingredient]]+=1
    return(featurefreq,featuresubset,cuisinedict)
#%%
def listodict(lis):
    dic=dict()
    for i,word in enumerate(lis):
        dic[word]=i
    return(dic)
#%%

def debugprint(string):
    print("DEBUG: " + str(string))

def printtrainingdatastats(examples):    
    featuresubset=np.empty(20,dtype=object)
    for i in range(len(featuresubset)):
        featuresubset[i]=set()    
    cuisinedict={}
    featureset = set()
    labels = set()
    examplecount = 0
    for example in examples:
        examplecount += 1
        if not(example["cuisine"] in labels):
            labels.add(example["cuisine"])
            cuisinedict[example["cuisine"]]=len(labels)-1
        for ingredient in example["ingredients"]:
            featuresubset[cuisinedict[example["cuisine"]]].add(ingredient)
    for dummy in featuresubset:
        featureset=featureset.union(dummy)
    print("Training Data Properties:")
    print("Number of Examples (Recipes):            " + str(examplecount))
    print("Number of Features (Unique Ingredients): " + str(len(featureset)))
    print("Number of Labels (Cuisines):             " + str(len(labels)))

    for cuisine in labels:
        print "Size of the",cuisine,"cuisine:",len(featuresubset[cuisinedict \
        [cuisine]])
    #
    comingredients=featuresubset[0]

    for subset in featuresubset:
        comingredients=set.intersection(comingredients,subset)
    debugprint(labels)
    print " the ingredients you find everywhere are:"
    print(comingredients)
    featurefreq,featuresubset,cuisinedict=completestat(examples)


def main():
    arguments = validateInput(sys.argv)
    algorithm,featurextraction,printtrainingstats = arguments
    # 1: Perceptron OVA, 2: Decision Tree  
    algorithms = { 1 : runperceptronOVA, 2 : rundecisiontree }
    algorithmnames = {1 : "Perceptron One vs. All", 2 : "Decision Tree"}


    # Read in the data file
    trainingdata = open("train.json")
    trainingexamples = json.load(trainingdata)
    
    if (printtrainingstats):
        printtrainingdatastats(trainingexamples)

    import cProfile, pstats, StringIO
    
    ## FRANCOIS - I think that eventually we might like a common 
    ## routine for extracting features from the training data.  For
    ## the moment, I'm extracting the features that I need for 
    ## Perceptron OVA in the algorithm routine itself.
        
    ##### BEGIN Algorithm Execution #####
    pr = cProfile.Profile()
    pr.enable()

    ## FRANCOIS - For the moment, I'm just passing in the training
    ## examples and getting the training accuracy.  Since we'll likely
    ## be terrible initially, I figured test statistics aren't all that
    ## helpful.  
    trainingaccuracy = algorithms[algorithm](trainingexamples)   

    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    print("ALGORITHM EXECUTION RUNTIME STATISTICS:")
    ps.print_stats()
    stats = s.getvalue()
    ##### END Algorithm Execution #####

    print("Algorithm:            " + algorithmnames[algorithm])
    print("Training Accuracy:    " + str(trainingaccuracy))

if __name__ == '__main__':
    main()



