import featureset
import json
import numpy as np
import jc_features as fc
import os
import nltk
import jc_means_500

class FSKMeansEuclidean500(featureset.FeatureSet):
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
            FSKMeansEuclidean500.word2index,FSKMeansEuclidean500.index2word = fc.getwordembeddingdictionaries(trainingexamples)
            wordembeddingdictionaries = [FSKMeansEuclidean500.word2index,FSKMeansEuclidean500.index2word]
            json.dump(wordembeddingdictionaries, wordembeddingdictfile)
            wordembeddingdictfile.flush()
            wordembeddingdictfile.close()
        else:
            wordembeddingdictfile = open("features/wordembeddingdictionaries.json")
            wordembeddingdictionaries = json.load(wordembeddingdictfile)
            FSKMeansEuclidean500.word2index,FSKMeansEuclidean500.index2word = wordembeddingdictionaries
            wordembeddingdictfile.close()
        if not os.path.exists("features/FSKMeansEuclidean500.json"):
            featurefile = open("features/FSKMeansEuclidean500.json",'wb')            
            cuisineindex = 0
            for example in trainingexamples:
                try: 
                    cindex = FSKMeansEuclidean500.cuisines[example["cuisine"]]                
                except KeyError:
                    cindex = cuisineindex
                    FSKMeansEuclidean500.cuisines[example["cuisine"]] = cindex                
                    FSKMeansEuclidean500.labels[cindex] = example["cuisine"]
                    cuisineindex += 1 
            ingredientlist = fc.getingredientlist(trainingexamples)            
            ingredientvectors = fc.getingredientvectors(FSKMeansEuclidean500.word2index, ingredientlist)
            means = [np.array(mean) for mean in jc_means_500.euclidean5]
            FSKMeansEuclidean500.clusterer = nltk.cluster.kmeans.KMeansClusterer(500,nltk.cluster.euclidean_distance,repeats = 1, initial_means = means)        
            
            for trainingexample in trainingexamples:
                FSKMeansEuclidean500.fmttrainingexamples.append(FSKMeansEuclidean500.formatexample(self, trainingexample))
            
            for testexample in testexamples:
                FSKMeansEuclidean500.fmttestexamples.append(FSKMeansEuclidean500.formatexample(self, testexample))
             
            featuredata = dict()
            featuredata["train"] = FSKMeansEuclidean500.fmttrainingexamples 
            featuredata["test"] = FSKMeansEuclidean500.fmttestexamples
            featuredata["labels"] = FSKMeansEuclidean500.labels
                                                      
            json.dump(featuredata,featurefile)            
            featurefile.flush()
            featurefile.close()
        else:
            featurefile = open("features/FSKMeansEuclidean500.json")            
            featuredata = json.load(featurefile)
            FSKMeansEuclidean500.fmttrainingexamples = featuredata["train"]
            FSKMeansEuclidean500.fmttestexamples = featuredata["test"] 
            templabels = featuredata["labels"]
            for key in templabels.keys():
                FSKMeansEuclidean500.labels[int(key)] = templabels[key]
            featurefile.close()        

    def formattedtrainingexamples(self):
        return FSKMeansEuclidean500.fmttrainingexamples 

    def formattedtestexamples(self):
        return FSKMeansEuclidean500.fmttestexamples    

    def formatexample(self,example):
        try:
            label = FSKMeansEuclidean500.cuisines[example["cuisine"]]
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
                cluster = FSKMeansEuclidean500.clusterdict[ingredient]
            except KeyError:
                ingredientvectors = fc.getingredientvectors(FSKMeansEuclidean500.word2index, [ingredient])                
                cluster = FSKMeansEuclidean500.clusterer.classify(ingredientvectors[0])
            featurevector.append(cluster)
        return [label,featurevector,id]

    def label(self, fmt_label):
        return FSKMeansEuclidean500.labels[fmt_label]

    def description(self):
        pass
        
    def labelfromexample(self,fmt_example):
        return FSKMeansEuclidean500.label(self,fmt_example[0])

    def formattedlabels(self):
        return FSKMeansEuclidean500.labels.keys()

    def formattedexamplevectorlength(self):
        return 500

    def idfromexample(self,example):
        return example[2]

