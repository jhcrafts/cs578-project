import featureset
import os
import json

class FSSpellCheckVectors(featureset.FeatureSet):
    """description of class"""
    cuisines = dict()
    labels = dict()    
    ingredients = dict()

    fmttrainingexamples = list()
    fmttestexamples = list()
    
    def extractfeatures(self,trainingexamples, testexamples):        
        if not os.path.exists("features/FSSpellCheckVectors.json"):            
            featurefile = open("features/FSSpellCheckVectors.json",'wb')            
            ingredientindex = 0 
            cuisineindex = 0
            for example in trainingexamples:
                try: 
                    cindex = FSSpellCheckVectors.cuisines[example["cuisine"]]                
                except KeyError:
                    cindex = cuisineindex
                    FSSpellCheckVectors.cuisines[example["cuisine"]] = cindex                
                    FSSpellCheckVectors.labels[cindex] = example["cuisine"]
                    cuisineindex += 1                            
                for ingredient in example["ingredients"]: 
                    try: 
                        iindex = FSSpellCheckVectors.ingredients[ingredient]
                    except KeyError:
                        iindex = ingredientindex
                        FSSpellCheckVectors.ingredients[ingredient] = iindex
                        ingredientindex += 1
            
            for trainingexample in trainingexamples:
                FSSpellCheckVectors.fmttrainingexamples.append(FSSpellCheckVectors.formatexample(self, trainingexample))
            
            for testexample in testexamples:
                FSSpellCheckVectors.fmttestexamples.append(FSSpellCheckVectors.formatexample(self, testexample))
             
            featuredata = dict()
            featuredata["train"] = FSSpellCheckVectors.fmttrainingexamples 
            featuredata["test"] = FSSpellCheckVectors.fmttestexamples
            featuredata["labels"] = FSSpellCheckVectors.labels
            FSSpellCheckVectors.vectorlength = len(FSSpellCheckVectors.ingredients.keys())
            featuredata["vectorlength"] = FSSpellCheckVectors.vectorlength
                                                      
            json.dump(featuredata,featurefile)            
            featurefile.flush()
            featurefile.close()
        else:
            featurefile = open("features/FSSpellCheckVectors.json")            
            featuredata = json.load(featurefile)
            FSSpellCheckVectors.fmttrainingexamples = featuredata["train"]
            FSSpellCheckVectors.fmttestexamples = featuredata["test"] 
            templabels = featuredata["labels"]
            FSSpellCheckVectors.vectorlength = featuredata["vectorlength"]
            for key in templabels.keys():
                FSSpellCheckVectors.labels[int(key)] = templabels[key]
            featurefile.close()        
       
    def formatexample(self,example):
        try:
            label = FSSpellCheckVectors.cuisines[example["cuisine"]]
        except KeyError:
            label = None
        try:
            id = FSSpellCheckVectors.cuisines[example["id"]]
        except KeyError:
            id = 0
        featurevector = list()
        for ingredient in example["ingredients"]:
            try:
                featurevector.append(FSSpellCheckVectors.ingredients[ingredient])
            except KeyError:
                pass   
        return [label,featurevector, id]
  
    def formattedtrainingexamples(self):
        return FSSpellCheckVectors.fmttrainingexamples 

    def formattedtestexamples(self):
        return FSSpellCheckVectors.fmttestexamples  
    
    def label(self, fmt_label):
        return FSSpellCheckVectors.labels[fmt_label]
      
    def labelfromexample(self,fmt_example):
        return self.label(fmt_example[0])

    def formattedlabels(self):
        return FSSpellCheckVectors.labels.keys()

    def formattedexamplevectorlength(self):
        return FSSpellCheckVectors.vectorlength

    def idfromexample(self,example):
        return example[2]
