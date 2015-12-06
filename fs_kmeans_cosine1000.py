import featureset
import json
import numpy as np
import jc_features as fc
import os
import nltk
import jc_means_cosine_1000

class FSKMeansCosine1000(featureset.FeatureSet):
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
            FSKMeansCosine1000.word2index,FSKMeansCosine1000.index2word = fc.getwordembeddingdictionaries(trainingexamples)
            wordembeddingdictionaries = [FSKMeansCosine1000.word2index,FSKMeansCosine1000.index2word]
            json.dump(wordembeddingdictionaries, wordembeddingdictfile)
            wordembeddingdictfile.flush()
            wordembeddingdictfile.close()
        else:
            wordembeddingdictfile = open("features/wordembeddingdictionaries.json")
            wordembeddingdictionaries = json.load(wordembeddingdictfile)
            FSKMeansCosine1000.word2index,FSKMeansCosine1000.index2word = wordembeddingdictionaries
            wordembeddingdictfile.close()
        if not os.path.exists("features/FSKMeansCosine1000.json"):
            featurefile = open("features/FSKMeansCosine1000.json",'wb')            
            cuisineindex = 0
            for example in trainingexamples:
                try: 
                    cindex = FSKMeansCosine1000.cuisines[example["cuisine"]]                
                except KeyError:
                    cindex = cuisineindex
                    FSKMeansCosine1000.cuisines[example["cuisine"]] = cindex                
                    FSKMeansCosine1000.labels[cindex] = example["cuisine"]
                    cuisineindex += 1 
            ingredientlist = fc.getingredientlist(trainingexamples)            
            ingredientvectors = fc.getingredientvectors(FSKMeansCosine1000.word2index, ingredientlist)
            means = [np.array(mean) for mean in jc_means_cosine_1000.cosine10]
            FSKMeansCosine1000.clusterer = nltk.cluster.kmeans.KMeansClusterer(1000,nltk.cluster.cosine_distance,repeats = 1, initial_means = means)        
            
            for trainingexample in trainingexamples:
                FSKMeansCosine1000.fmttrainingexamples.append(FSKMeansCosine1000.formatexample(self, trainingexample))
            
            for testexample in testexamples:
                FSKMeansCosine1000.fmttestexamples.append(FSKMeansCosine1000.formatexample(self, testexample))
             
            featuredata = dict()
            featuredata["train"] = FSKMeansCosine1000.fmttrainingexamples 
            featuredata["test"] = FSKMeansCosine1000.fmttestexamples
            featuredata["labels"] = FSKMeansCosine1000.labels
                                                      
            json.dump(featuredata,featurefile)            
            featurefile.flush()
            featurefile.close()
        else:
            featurefile = open("features/FSKMeansCosine1000.json")            
            featuredata = json.load(featurefile)
            FSKMeansCosine1000.fmttrainingexamples = featuredata["train"]
            FSKMeansCosine1000.fmttestexamples = featuredata["test"] 
            templabels = featuredata["labels"]
            for key in templabels.keys():
                FSKMeansCosine1000.labels[int(key)] = templabels[key]
            featurefile.close()        

    def formattedtrainingexamples(self):
        return FSKMeansCosine1000.fmttrainingexamples 

    def formattedtestexamples(self):
        return FSKMeansCosine1000.fmttestexamples    

    def formatexample(self,example):
        try:
            label = FSKMeansCosine1000.cuisines[example["cuisine"]]
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
                cluster = FSKMeansCosine1000.clusterdict[ingredient]
            except KeyError:
                ingredientvectors = fc.getingredientvectors(FSKMeansCosine1000.word2index, [ingredient])                
                cluster = FSKMeansCosine1000.clusterer.classify(ingredientvectors[0])
            featurevector.append(cluster)
        return [label,featurevector,id]

    def label(self, fmt_label):
        return FSKMeansCosine1000.labels[fmt_label]

    def description(self):
        pass
        
    def labelfromexample(self,fmt_example):
        return FSKMeansCosine1000.label(self,fmt_example[0])

    def formattedlabels(self):
        return FSKMeansCosine1000.labels.keys()

    def formattedexamplevectorlength(self):
        return 1000

    def idfromexample(self,example):
        return example[2]

