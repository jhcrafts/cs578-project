import featureset
import json
import numpy as np
import jc_features as fc
import os
import nltk
import jc_means_1000

class FSKMeansEuclidean1000(featureset.FeatureSet):
    """description of class"""
    cuisines = dict()
    labels = dict()
    ingredients = dict()
    clusterer = None
    word2index = None
    index2word = None  
    clustermeans = None
    clusterdict = dict()

    fmttrainingexamples = list()
    fmttestexamples = list()

    def extractfeatures(self, trainingexamples, testexamples):
        if not os.path.exists("features/wordembeddingdictionaries.json"):
            wordembeddingdictfile = open("features/wordembeddingdictionaries.json",'wb')
            FSKMeansEuclidean1000.word2index,FSKMeansEuclidean1000.index2word = fc.getwordembeddingdictionaries(trainingexamples)
            wordembeddingdictionaries = [FSKMeansEuclidean1000.word2index,FSKMeansEuclidean1000.index2word]
            json.dump(wordembeddingdictionaries, wordembeddingdictfile)
            wordembeddingdictfile.flush()
            wordembeddingdictfile.close()
        else:
            wordembeddingdictfile = open("features/wordembeddingdictionaries.json")
            wordembeddingdictionaries = json.load(wordembeddingdictfile)
            FSKMeansEuclidean1000.word2index,FSKMeansEuclidean1000.index2word = wordembeddingdictionaries
            wordembeddingdictfile.close()
        if not os.path.exists("features/FSKMeansEuclidean1000.json"):
            featurefile = open("features/FSKMeansEuclidean1000.json",'wb')            
            cuisineindex = 0
            for example in trainingexamples:
                try: 
                    cindex = FSKMeansEuclidean1000.cuisines[example["cuisine"]]                
                except KeyError:
                    cindex = cuisineindex
                    FSKMeansEuclidean1000.cuisines[example["cuisine"]] = cindex                
                    FSKMeansEuclidean1000.labels[cindex] = example["cuisine"]
                    cuisineindex += 1 
            ingredientlist = fc.getingredientlist(trainingexamples)            
            ingredientvectors = fc.getingredientvectors(FSKMeansEuclidean1000.word2index, ingredientlist)
            means = [np.array(mean) for mean in jc_means_1000.euclidean10]
            FSKMeansEuclidean1000.clusterer = nltk.cluster.kmeans.KMeansClusterer(1000,nltk.cluster.euclidean_distance,repeats = 1, initial_means = means)        
            
            for trainingexample in trainingexamples:
                FSKMeansEuclidean1000.fmttrainingexamples.append(FSKMeansEuclidean1000.formatexample(self, trainingexample))
            
            for testexample in testexamples:
                FSKMeansEuclidean1000.fmttestexamples.append(FSKMeansEuclidean1000.formatexample(self, testexample))
             
            featuredata = dict()
            featuredata["train"] = FSKMeansEuclidean1000.fmttrainingexamples 
            featuredata["test"] = FSKMeansEuclidean1000.fmttestexamples
            featuredata["labels"] = FSKMeansEuclidean1000.labels
                                                      
            json.dump(featuredata,featurefile)            
            featurefile.flush()
            featurefile.close()
        else:
            featurefile = open("features/FSKMeansEuclidean1000.json")            
            featuredata = json.load(featurefile)
            FSKMeansEuclidean1000.fmttrainingexamples = featuredata["train"]
            FSKMeansEuclidean1000.fmttestexamples = featuredata["test"] 
            templabels = featuredata["labels"]
            for key in templabels.keys():
                FSKMeansEuclidean1000.labels[int(key)] = templabels[key]
            featurefile.close()        

    def formattedtrainingexamples(self):
        return FSKMeansEuclidean1000.fmttrainingexamples 

    def formattedtestexamples(self):
        return FSKMeansEuclidean1000.fmttestexamples    

    def formatexample(self,example):
        try:
            label = FSKMeansEuclidean1000.cuisines[example["cuisine"]]
        except KeyError:
            label = None
        try:
            id = example["id"]
        except KeyError:
            id = 0
        featurevector = list()
        for ingredient in example["ingredients"]:
            cluster = 0
            try:
                cluster = FSKMeansEuclidean1000.clusterdict[ingredient]
            except KeyError:
                ingredientvectors = fc.getingredientvectors(FSKMeansEuclidean1000.word2index, [ingredient])                
                cluster = FSKMeansEuclidean1000.clusterer.classify(ingredientvectors[0])
            featurevector.append(cluster)
        return [label,featurevector,id]

    def label(self, fmt_label):
        return FSKMeansEuclidean1000.labels[fmt_label]

    def description(self):
        pass
        
    def labelfromexample(self,fmt_example):
        return FSKMeansEuclidean1000.label(self,fmt_example[0])

    def formattedlabels(self):
        return FSKMeansEuclidean1000.labels.keys()

    def formattedexamplevectorlength(self):
        return 1000

    def idfromexample(self,example):
        return example[2]