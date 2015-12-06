import featureset
import json
import numpy as np
import jc_features as fc
import os
import nltk
import jc_means_100

class FSKMeansEuclidean100(featureset.FeatureSet):
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
            FSKMeansEuclidean100.word2index,FSKMeansEuclidean100.index2word = fc.getwordembeddingdictionaries(trainingexamples)
            wordembeddingdictionaries = [FSKMeansEuclidean100.word2index,FSKMeansEuclidean100.index2word]
            json.dump(wordembeddingdictionaries, wordembeddingdictfile)
            wordembeddingdictfile.flush()
            wordembeddingdictfile.close()
        else:
            wordembeddingdictfile = open("features/wordembeddingdictionaries.json")
            wordembeddingdictionaries = json.load(wordembeddingdictfile)
            FSKMeansEuclidean100.word2index,FSKMeansEuclidean100.index2word = wordembeddingdictionaries
            wordembeddingdictfile.close()
        if not os.path.exists("features/FSKMeansEuclidean100.json"):
            featurefile = open("features/FSKMeansEuclidean100.json",'wb')            
            cuisineindex = 0
            for example in trainingexamples:
                try: 
                    cindex = FSKMeansEuclidean100.cuisines[example["cuisine"]]                
                except KeyError:
                    cindex = cuisineindex
                    FSKMeansEuclidean100.cuisines[example["cuisine"]] = cindex                
                    FSKMeansEuclidean100.labels[cindex] = example["cuisine"]
                    cuisineindex += 1 
            ingredientlist = fc.getingredientlist(trainingexamples)            
            ingredientvectors = fc.getingredientvectors(FSKMeansEuclidean100.word2index, ingredientlist)
            means = [np.array(mean) for mean in jc_means_100.euclidean1]
            FSKMeansEuclidean100.clusterer = nltk.cluster.kmeans.KMeansClusterer(100,nltk.cluster.euclidean_distance,repeats = 1, initial_means = means)        
            
            for trainingexample in trainingexamples:
                FSKMeansEuclidean100.fmttrainingexamples.append(FSKMeansEuclidean100.formatexample(self, trainingexample))
            
            for testexample in testexamples:
                FSKMeansEuclidean100.fmttestexamples.append(FSKMeansEuclidean100.formatexample(self, testexample))
             
            featuredata = dict()
            featuredata["train"] = FSKMeansEuclidean100.fmttrainingexamples 
            featuredata["test"] = FSKMeansEuclidean100.fmttestexamples
            featuredata["labels"] = FSKMeansEuclidean100.labels
                                                      
            json.dump(featuredata,featurefile)            
            featurefile.flush()
            featurefile.close()
        else:
            featurefile = open("features/FSKMeansEuclidean100.json")            
            featuredata = json.load(featurefile)
            FSKMeansEuclidean100.fmttrainingexamples = featuredata["train"]
            FSKMeansEuclidean100.fmttestexamples = featuredata["test"] 
            templabels = featuredata["labels"]
            for key in templabels.keys():
                FSKMeansEuclidean100.labels[int(key)] = templabels[key]
            featurefile.close()        

    def formattedtrainingexamples(self):
        return FSKMeansEuclidean100.fmttrainingexamples 

    def formattedtestexamples(self):
        return FSKMeansEuclidean100.fmttestexamples    

    def formatexample(self,example):
        try:
            label = FSKMeansEuclidean100.cuisines[example["cuisine"]]
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
                cluster = FSKMeansEuclidean100.clusterdict[ingredient]
            except KeyError:
                ingredientvectors = fc.getingredientvectors(FSKMeansEuclidean100.word2index, [ingredient])                
                cluster = FSKMeansEuclidean100.clusterer.classify(ingredientvectors[0])
            featurevector.append(cluster)
        return [label,featurevector,id]

    def label(self, fmt_label):
        return FSKMeansEuclidean100.labels[fmt_label]

    def description(self):
        pass
        
    def labelfromexample(self,fmt_example):
        return FSKMeansEuclidean100.label(self,fmt_example[0])

    def formattedlabels(self):
        return FSKMeansEuclidean100.labels.keys()

    def formattedexamplevectorlength(self):
        return 100

    def idfromexample(self,example):
        return example[2]