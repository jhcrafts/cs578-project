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

import alg_perceptronOVA as PerceptronOVA
import alg_decisiontree as DecisionTree

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

def main():
    arguments = validateInput(sys.argv)
    algorithm,featurextraction,printtrainingstats = arguments
    # 1: Perceptron OVA, 2: Decision Tree  
    algorithms = { 
        1 : PerceptronOVA.AlgPerceptronOVA(), 
        2 : DecisionTree.AlgDecisionTree() 
        }

    classifier = algorithms[algorithm]

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

    ## FRANCOIS - I've updated this to train the classifier and then we can run
    ## whatever statistics we want by using the predict routine of the base
    ## class.  I figured this would be a bit more portable and easy to augment.  
    classifier.train(trainingexamples)   

    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    print("CLASSIFIER TRAINING RUNTIME STATISTICS:")
    ps.print_stats()
    stats = s.getvalue()
    ##### END Algorithm Execution #####

    print("Algorithm:            " + classifier.name())
    print("Training Accuracy:    " + str(0.0))

if __name__ == '__main__':
    main()



