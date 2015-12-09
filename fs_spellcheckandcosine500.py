import featureset
import os
import json
import copy

class FSSpellCheckCosine500(featureset.FeatureSet):
    """description of class"""
    cuisines = dict()
    labels = dict()    
    ingredients = dict()
    ingredienttupledict = dict()
    fmttrainingexamples = list()
    fmttestexamples = list()
    vectorlength = 0
    
    def extractfeatures(self,trainingexamples, testexamples):
        if not os.path.exists("features/FSSpellCheckCosine500.json"):
            featurefile = open("features/FSSpellCheckCosine500.json",'wb')            
            
            spellcheckfile = open("features/FSSpellCheckVectors.json")
            spellcheckfeaturedata = json.load(spellcheckfile)

            spellcheckfmttrainingexamples = spellcheckfeaturedata["train"]
            spellcheckfmttestexamples = spellcheckfeaturedata["test"] 
            spellcheckvectorlength = spellcheckfeaturedata["vectorlength"]
            FSSpellCheckCosine500.labels = spellcheckfeaturedata["labels"]

            cosinefile = open("features/FSKMeansCosine500.json")
            cosinefeaturedata = json.load(cosinefile)
            cosinefmttrainingexamples = cosinefeaturedata["train"]
            cosinefmttestexamples = cosinefeaturedata["test"] 
            cosinevectorlength = 500
                            
            for example in spellcheckfmttrainingexamples:
                for i in range(len(example[1])):
                    example[1][i] += cosinevectorlength
            for example in spellcheckfmttestexamples:
                for i in range(len(example[1])):
                    example[1][i] += cosinevectorlength
                    
            for i in range(len(spellcheckfmttrainingexamples)):
                featurevector = cosinefmttrainingexamples[i][1] + spellcheckfmttrainingexamples[i][1]
                FSSpellCheckCosine500.fmttrainingexamples.append([spellcheckfmttrainingexamples[i][0],featurevector,spellcheckfmttrainingexamples[i][2]])

            for i in range(len(spellcheckfmttestexamples)):
                featurevector = cosinefmttestexamples[i][1] + spellcheckfmttestexamples[i][1]
                FSSpellCheckCosine500.fmttestexamples.append([spellcheckfmttestexamples[i][0],featurevector,spellcheckfmttestexamples[i][2]])

            FSSpellCheckCosine500.vectorlength = cosinevectorlength + spellcheckvectorlength

            featuredata = dict()
            featuredata["train"] = FSSpellCheckCosine500.fmttrainingexamples 
            featuredata["test"] = FSSpellCheckCosine500.fmttestexamples
            featuredata["labels"] = FSSpellCheckCosine500.labels
            featuredata["vectorlength"] = FSSpellCheckCosine500.vectorlength
                                                      
            json.dump(featuredata,featurefile)            
            featurefile.flush()
            featurefile.close()
        else:
            featurefile = open("features/FSSpellCheckCosine500.json")            
            featuredata = json.load(featurefile)
            FSSpellCheckCosine500.fmttrainingexamples = featuredata["train"]
            FSSpellCheckCosine500.fmttestexamples = featuredata["test"] 
            templabels = featuredata["labels"]
            for key in templabels.keys():
                FSSpellCheckCosine500.labels[int(key)] = templabels[key]            
            FSSpellCheckCosine500.vectorlength = featuredata["vectorlength"]
            featurefile.close()        
       
    def formatexample(self,example):
        try:
            label = FSSpellCheckCosine500.cuisines[example["cuisine"]]
        except KeyError:
            label = None
        
        featurevector = list()
        for ingredient in example["ingredients"]:
            try:
                featurevector.append(FSSpellCheckCosine500.ingredients[ingredient])
            except KeyError:
                pass   
        return [label,featurevector, id]
  
    def formattedtrainingexamples(self):
        return FSSpellCheckCosine500.fmttrainingexamples 

    def formattedtestexamples(self):
        return FSSpellCheckCosine500.fmttestexamples  
    
    def label(self, fmt_label):
        return FSSpellCheckCosine500.labels[fmt_label]
      
    def labelfromexample(self,fmt_example):
        return self.label(fmt_example[0])

    def formattedlabels(self):
        return FSSpellCheckCosine500.labels.keys()

    def formattedexamplevectorlength(self):
        return FSSpellCheckCosine500.vectorlength

    def idfromexample(self,example):
        return example[2]



