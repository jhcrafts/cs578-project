import featureset
import json
import numpy as np
import jc_features as fc
import os
import nltk
import jc_means_cosine_500

class FSKMeansCosine500(featureset.FeatureSet):
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
            FSKMeansCosine500.word2index,FSKMeansCosine500.index2word = fc.getwordembeddingdictionaries(trainingexamples)
            wordembeddingdictionaries = [FSKMeansCosine500.word2index,FSKMeansCosine500.index2word]
            json.dump(wordembeddingdictionaries, wordembeddingdictfile)
            wordembeddingdictfile.flush()
            wordembeddingdictfile.close()
        else:
            wordembeddingdictfile = open("features/wordembeddingdictionaries.json")
            wordembeddingdictionaries = json.load(wordembeddingdictfile)
            FSKMeansCosine500.word2index,FSKMeansCosine500.index2word = wordembeddingdictionaries
            wordembeddingdictfile.close()
        if not os.path.exists("features/FSKMeansCosine500.json"):
            featurefile = open("features/FSKMeansCosine500.json",'wb')            
            cuisineindex = 0
            for example in trainingexamples:
                try: 
                    cindex = FSKMeansCosine500.cuisines[example["cuisine"]]                
                except KeyError:
                    cindex = cuisineindex
                    FSKMeansCosine500.cuisines[example["cuisine"]] = cindex                
                    FSKMeansCosine500.labels[cindex] = example["cuisine"]
                    cuisineindex += 1 
            ingredientlist = fc.getingredientlist(trainingexamples)            
            ingredientvectors = fc.getingredientvectors(FSKMeansCosine500.word2index, ingredientlist)
            means = [np.array(mean) for mean in jc_means_cosine_500.cosine5]
            FSKMeansCosine500.clusterer = nltk.cluster.kmeans.KMeansClusterer(500,nltk.cluster.cosine_distance,repeats = 1, initial_means = means)        
            
            for trainingexample in trainingexamples:
                FSKMeansCosine500.fmttrainingexamples.append(FSKMeansCosine500.formatexample(self, trainingexample))
            
            for testexample in testexamples:
                FSKMeansCosine500.fmttestexamples.append(FSKMeansCosine500.formatexample(self, testexample))
             
            featuredata = dict()
            featuredata["train"] = FSKMeansCosine500.fmttrainingexamples 
            featuredata["test"] = FSKMeansCosine500.fmttestexamples
            featuredata["labels"] = FSKMeansCosine500.labels
                                                      
            json.dump(featuredata,featurefile)            
            featurefile.flush()
            featurefile.close()
        else:
            featurefile = open("features/FSKMeansCosine500.json")            
            featuredata = json.load(featurefile)
            FSKMeansCosine500.fmttrainingexamples = featuredata["train"]
            FSKMeansCosine500.fmttestexamples = featuredata["test"] 
            templabels = featuredata["labels"]
            for key in templabels.keys():
                FSKMeansCosine500.labels[int(key)] = templabels[key]
            featurefile.close()        

    def formattedtrainingexamples(self):
        return FSKMeansCosine500.fmttrainingexamples 

    def formattedtestexamples(self):
        return FSKMeansCosine500.fmttestexamples    

    def formatexample(self,example):
        try:
            label = FSKMeansCosine500.cuisines[example["cuisine"]]
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
                cluster = FSKMeansCosine500.clusterdict[ingredient]
            except KeyError:
                ingredientvectors = fc.getingredientvectors(FSKMeansCosine500.word2index, [ingredient])                
                cluster = FSKMeansCosine500.clusterer.classify(ingredientvectors[0])
            featurevector.append(cluster)
        return [label,featurevector,id]

    def label(self, fmt_label):
        return FSKMeansCosine500.labels[fmt_label]

    def description(self):
        pass
        
    def labelfromexample(self,fmt_example):
        return FSKMeansCosine500.label(self,fmt_example[0])

    def formattedlabels(self):
        return FSKMeansCosine500.labels.keys()

    def formattedexamplevectorlength(self):
        return 500

    def idfromexample(self,example):
        return example[2]

