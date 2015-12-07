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
import alg_gradientdescentOVA as GradientDescentOVA

import fs_rawvectors as FeatureSetRawVectors
import fs_spellcheckvectors as FeatureSetSpellCheckVectors
import fs_kmeans_euclidean100 as FeatureSetKMeansEuclidean100
import fs_kmeans_euclidean500 as FeatureSetKMeansEuclidean500
import fs_kmeans_euclidean1000 as FeatureSetKMeansEuclidean1000
import fs_kmeans_cosine100 as FeatureSetKMeansCosine100
import fs_kmeans_cosine500 as FeatureSetKMeansCosine500
import fs_kmeans_cosine1000 as FeatureSetKMeansCosine1000
import fs_topingredientbigrams as FeatureSetTopBigrams
import fs_topingredienttrigrams as FeatureSetTopTrigrams

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

    algorithm = 1 # 1: Perceptron OVA, 2: Decision Tree, 3: Gradient Descent OVA    
    feature = 9 #1 Raw Feature Vectors
    iterations = 10 # 10 iterations by default
    forcefeatureextraction = False
    printtrainingdatastats = False
    description = 'default'
    submissionname = 'Kmeans500EuclideanPerceptronOVA40.csv'    

    if '-a' in args_map:
      algorithm = int(args_map['-a'])    
    if '-f' in args_map:
      feature = int(args_map['-f'])
    if '-i' in args_map:
      iterations = int(args_map['-i'])
    if '-p' in args_map:
      printtrainingdatastats = True
    if '-s' in args_map:
      description = args_map['-s']
    if '-n' in args_map:
      submissionname = args_map['-n']

    assert algorithm in [1, 2, 3]
    assert feature in [1,2,3,4,5,6,7,8,9,10]

    return [algorithm, feature, iterations, printtrainingdatastats, description, submissionname]

def main():
    arguments = validateInput(sys.argv)
    algorithm,feature,iterations,printtrainingstats, description, submissionname = arguments
    # 1: Perceptron OVA, 2: Decision Tree  
    algorithms = { 
        1 : PerceptronOVA.AlgPerceptronOVA(), 
        2 : DecisionTree.AlgDecisionTree(),
        3 : GradientDescentOVA.AlgGradientDescentOVA() 
        }

    featuresets = { 
        1 : FeatureSetRawVectors.FSRawVectors(),
        2 : FeatureSetSpellCheckVectors.FSSpellCheckVectors(),
        3 : FeatureSetKMeansEuclidean100.FSKMeansEuclidean100(),
        4 : FeatureSetKMeansEuclidean500.FSKMeansEuclidean500(),
        5 : FeatureSetKMeansEuclidean1000.FSKMeansEuclidean1000(),
        6 : FeatureSetKMeansCosine100.FSKMeansCosine100(),
        7 : FeatureSetKMeansCosine500.FSKMeansCosine500(),
        8 : FeatureSetKMeansCosine1000.FSKMeansCosine1000(),
        9 : FeatureSetTopBigrams.FSTopIngredientBigramVectors(),
        10: FeatureSetTopTrigrams.FSTopIngredientTrigramVectors()
        }

    classifier = algorithms[algorithm]
    featureset = featuresets[feature]

    import jc_features as fc

    import os.path
    # Read in the data files
    if not os.path.exists("cleantrain.json"):
        trainingdata = open("train.json")
        trainingexamples = json.load(trainingdata)
        cleantrainingdata = open('cleantrain.json','wb')
        cleantrainingexamples = fc.cleanfeatureset(trainingexamples)
        json.dump(cleantrainingexamples,cleantrainingdata)
        cleantrainingdata.flush()
        cleantrainingdata.close()
        trainingdata.close()
   
    if not os.path.exists("cleantest.json"):
        testdata = open("test.json")
        testexamples = json.load(testdata)    
        cleantestdata = open('cleantest.json','wb')
        cleantestexamples = fc.cleanfeatureset(testexamples)
        json.dump(cleantestexamples,cleantestdata)
        cleantestdata.flush()
        cleantestdata.close()
        testdata.close()

    trainingdata = open("cleantrain.json")
    trainingexamples = json.load(trainingdata)

    # generation of k-means classifiers for later use####
    #import nltk
    #word2index,index2word = fc.getwordembeddingdictionaries(trainingexamples)
    #ingredients = fc.getingredientlist(trainingexamples)
    #vectors = fc.getingredientvectors(word2index, ingredients)    
    #try:            
    #    clusterer = fc.getcosineingredientclusterer(vectors, 1000, 5)
    #    print("cosine clusterer data:")    
    #    print("means: " + str(np.array(clusterer.means()).tolist()))
    #    fc.testclusterer(clusterer,vectors,index2word)            
    #except:
    #    print("exception on cluster generation")
    #return

    testdata = open("cleantest.json")
    testexamples = json.load(testdata)   

    if (printtrainingstats):
        printtrainingdatastats(trainingexamples)

    import cProfile, pstats, StringIO
    
    ## FRANCOIS - I think that eventually we might like a common 
    ## routine for extracting features from the training data.  For
    ## the moment, I'm extracting the features that I need for 
    ## Perceptron OVA in the algorithm routine itself.   

    ## First, let the classifier extract features/labels from the
    ## Training examples
    featureset.extractfeatures(trainingexamples, testexamples)   
    
    ##### BEGIN Algorithm Execution #####
    pr = cProfile.Profile()
    pr.enable()
    ## Time to train the classifier on the training examples
    classifier.train(featureset.formattedtrainingexamples(), featureset.formattedexamplevectorlength(), featureset.formattedlabels(), iterations)
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    print("CLASSIFIER TRAINING RUNTIME STATISTICS:")
    ps.print_stats()
    stats = s.getvalue()
    ##### END Algorithm Execution #####        
    
    ## Let's generate some statistics on the training data
    correct = 0.0
    total = 0.0
    for example in featureset.formattedtrainingexamples():
        total += 1
        if (featureset.label(classifier.predict(example)) == featureset.labelfromexample(example)):
                correct += 1
    trainingaccuracy = correct/total

    ## Let's create a test data submission for Kaggle    
    testsubmission = open(submissionname,'wb')
    submissionwriter = csv.writer(testsubmission)
    submissionwriter.writerow(["id","cuisine"])    
    for example in featureset.formattedtestexamples():
        fmt_label = classifier.predict(example)        
        submissionwriter.writerow([featureset.idfromexample(example), featureset.label(fmt_label)])
    testsubmission.close()

    print("Algorithm:            " + classifier.name())
    print("Training Accuracy:    " + str(trainingaccuracy))

if __name__ == '__main__':
    main()



