import featureset
import json
import numpy as np
import jc_features as fc
import os
import nltk
import jc_means_cosine_100

class FSKMeansCosine100(featureset.FeatureSet):
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
            FSKMeansCosine100.word2index,FSKMeansCosine100.index2word = fc.getwordembeddingdictionaries(trainingexamples)
            wordembeddingdictionaries = [FSKMeansCosine100.word2index,FSKMeansCosine100.index2word]
            json.dump(wordembeddingdictionaries, wordembeddingdictfile)
            wordembeddingdictfile.flush()
            wordembeddingdictfile.close()
        else:
            wordembeddingdictfile = open("features/wordembeddingdictionaries.json")
            wordembeddingdictionaries = json.load(wordembeddingdictfile)
            FSKMeansCosine100.word2index,FSKMeansCosine100.index2word = wordembeddingdictionaries
            wordembeddingdictfile.close()
        if not os.path.exists("features/FSKMeansCosine100.json"):
            featurefile = open("features/FSKMeansCosine100.json",'wb')            
            cuisineindex = 0
            for example in trainingexamples:
                try: 
                    cindex = FSKMeansCosine100.cuisines[example["cuisine"]]                
                except KeyError:
                    cindex = cuisineindex
                    FSKMeansCosine100.cuisines[example["cuisine"]] = cindex                
                    FSKMeansCosine100.labels[cindex] = example["cuisine"]
                    cuisineindex += 1 
            ingredientlist = fc.getingredientlist(trainingexamples)            
            ingredientvectors = fc.getingredientvectors(FSKMeansCosine100.word2index, ingredientlist)
            means = [np.array(mean) for mean in jc_means_cosine_100.cosine1]
            FSKMeansCosine100.clusterer = nltk.cluster.kmeans.KMeansClusterer(100,nltk.cluster.cosine_distance,repeats = 1, initial_means = means)        
            
            for trainingexample in trainingexamples:
                FSKMeansCosine100.fmttrainingexamples.append(FSKMeansCosine100.formatexample(self, trainingexample))
            
            for testexample in testexamples:
                FSKMeansCosine100.fmttestexamples.append(FSKMeansCosine100.formatexample(self, testexample))
             
            featuredata = dict()
            featuredata["train"] = FSKMeansCosine100.fmttrainingexamples 
            featuredata["test"] = FSKMeansCosine100.fmttestexamples
            featuredata["labels"] = FSKMeansCosine100.labels
                                                      
            json.dump(featuredata,featurefile)            
            featurefile.flush()
            featurefile.close()
        else:
            featurefile = open("features/FSKMeansCosine100.json")            
            featuredata = json.load(featurefile)
            FSKMeansCosine100.fmttrainingexamples = featuredata["train"]
            FSKMeansCosine100.fmttestexamples = featuredata["test"] 
            templabels = featuredata["labels"]
            for key in templabels.keys():
                FSKMeansCosine100.labels[int(key)] = templabels[key]
            featurefile.close()        

    def formattedtrainingexamples(self):
        return FSKMeansCosine100.fmttrainingexamples 

    def formattedtestexamples(self):
        return FSKMeansCosine100.fmttestexamples    

    def formatexample(self,example):
        try:
            label = FSKMeansCosine100.cuisines[example["cuisine"]]
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
                cluster = FSKMeansCosine100.clusterdict[ingredient]
            except KeyError:
                ingredientvectors = fc.getingredientvectors(FSKMeansCosine100.word2index, [ingredient])                
                cluster = FSKMeansCosine100.clusterer.classify(ingredientvectors[0])
            featurevector.append(cluster)
        return [label,featurevector,id]

    def label(self, fmt_label):
        return FSKMeansCosine100.labels[fmt_label]

    def description(self):
        pass
        
    def labelfromexample(self,fmt_example):
        return FSKMeansCosine100.label(self,fmt_example[0])

    def formattedlabels(self):
        return FSKMeansCosine100.labels.keys()

    def formattedexamplevectorlength(self):
        return 100

    def idfromexample(self,example):
        return example[2]

