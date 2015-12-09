import featureset
import os
import json
import copy

class FSSpellCheckTrigrams(featureset.FeatureSet):
    """description of class"""
    cuisines = dict()
    labels = dict()    
    ingredients = dict()
    ingredienttupledict = dict()
    fmttrainingexamples = list()
    fmttestexamples = list()
    vectorlength = 0
    
    def extractfeatures(self,trainingexamples, testexamples):
        if not os.path.exists("features/FSSpellCheckTrigrams.json"):
            featurefile = open("features/FSSpellCheckTrigrams.json",'wb')            
            
            spellcheckfile = open("features/FSSpellCheckVectors.json")
            spellcheckfeaturedata = json.load(spellcheckfile)

            spellcheckfmttrainingexamples = spellcheckfeaturedata["train"]
            spellcheckfmttestexamples = spellcheckfeaturedata["test"] 
            spellcheckvectorlength = spellcheckfeaturedata["vectorlength"]
            FSSpellCheckTrigrams.labels = spellcheckfeaturedata["labels"]

            trigramfile = open("features/FSTopIngredientTrigramVectors.json")
            trigramfeaturedata = json.load(trigramfile)
            trigramfmttrainingexamples = trigramfeaturedata["train"]
            trigramfmttestexamples = trigramfeaturedata["test"] 
            trigramvectorlength = trigramfeaturedata["vectorlength"]
                            
            for example in spellcheckfmttrainingexamples:
                for i in range(len(example[1])):
                    example[1][i] += trigramvectorlength
            for example in spellcheckfmttestexamples:
                for i in range(len(example[1])):
                    example[1][i] += trigramvectorlength
                    
            for i in range(len(spellcheckfmttrainingexamples)):
                featurevector = trigramfmttrainingexamples[i][1] + spellcheckfmttrainingexamples[i][1]
                FSSpellCheckTrigrams.fmttrainingexamples.append([spellcheckfmttrainingexamples[i][0],featurevector,spellcheckfmttrainingexamples[i][2]])

            for i in range(len(spellcheckfmttestexamples)):
                featurevector = trigramfmttestexamples[i][1] + spellcheckfmttestexamples[i][1]
                FSSpellCheckTrigrams.fmttestexamples.append([spellcheckfmttestexamples[i][0],featurevector,spellcheckfmttestexamples[i][2]])

            FSSpellCheckTrigrams.vectorlength = trigramvectorlength + spellcheckvectorlength

            featuredata = dict()
            featuredata["train"] = FSSpellCheckTrigrams.fmttrainingexamples 
            featuredata["test"] = FSSpellCheckTrigrams.fmttestexamples
            featuredata["labels"] = FSSpellCheckTrigrams.labels
            featuredata["vectorlength"] = FSSpellCheckTrigrams.vectorlength
                                                      
            json.dump(featuredata,featurefile)            
            featurefile.flush()
            featurefile.close()
        else:
            featurefile = open("features/FSSpellCheckTrigrams.json")            
            featuredata = json.load(featurefile)
            FSSpellCheckTrigrams.fmttrainingexamples = featuredata["train"]
            FSSpellCheckTrigrams.fmttestexamples = featuredata["test"] 
            templabels = featuredata["labels"]
            for key in templabels.keys():
                FSSpellCheckTrigrams.labels[int(key)] = templabels[key]            
            FSSpellCheckTrigrams.vectorlength = featuredata["vectorlength"]
            featurefile.close()        
       
    def formatexample(self,example):
        try:
            label = FSSpellCheckTrigrams.cuisines[example["cuisine"]]
        except KeyError:
            label = None
        
        featurevector = list()
        for ingredient in example["ingredients"]:
            try:
                featurevector.append(FSSpellCheckTrigrams.ingredients[ingredient])
            except KeyError:
                pass   
        return [label,featurevector, id]
  
    def formattedtrainingexamples(self):
        return FSSpellCheckTrigrams.fmttrainingexamples 

    def formattedtestexamples(self):
        return FSSpellCheckTrigrams.fmttestexamples  
    
    def label(self, fmt_label):
        return FSSpellCheckTrigrams.labels[fmt_label]
      
    def labelfromexample(self,fmt_example):
        return self.label(fmt_example[0])

    def formattedlabels(self):
        return FSSpellCheckTrigrams.labels.keys()

    def formattedexamplevectorlength(self):
        return FSSpellCheckTrigrams.vectorlength

    def idfromexample(self,example):
        return example[2]



