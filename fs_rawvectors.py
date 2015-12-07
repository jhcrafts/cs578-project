import featureset
import os
import json

class FSRawVectors(featureset.FeatureSet):
    """description of class"""
    cuisines = dict()
    labels = dict()    
    ingredients = dict()
    vectorlength = 0
    fmttrainingexamples = list()
    fmttestexamples = list()
    
    def extractfeatures(self,trainingexamples, testexamples):
        if not os.path.exists("features/FSRawVectors.json"):
            trainingdata = open("train.json")
            realtrainingexamples = json.load(trainingdata)
            trainingdata.close()
            testdata = open("test.json")
            realtestexamples = json.load(testdata)    
            testdata.close()
            featurefile = open("features/FSRawVectors.json",'wb')            
            ingredientindex = 0 
            cuisineindex = 0
            for example in realtrainingexamples:
                try: 
                    cindex = FSRawVectors.cuisines[example["cuisine"]]                
                except KeyError:
                    cindex = cuisineindex
                    FSRawVectors.cuisines[example["cuisine"]] = cindex                
                    FSRawVectors.labels[cindex] = example["cuisine"]
                    cuisineindex += 1                            
                for ingredient in example["ingredients"]: 
                    try: 
                        iindex = FSRawVectors.ingredients[ingredient]
                    except KeyError:
                        iindex = ingredientindex
                        FSRawVectors.ingredients[ingredient] = iindex
                        ingredientindex += 1
            
            for trainingexample in realtrainingexamples:
                FSRawVectors.fmttrainingexamples.append(FSRawVectors.formatexample(self, trainingexample))
            
            for testexample in realtestexamples:
                FSRawVectors.fmttestexamples.append(FSRawVectors.formatexample(self, testexample))
             
            featuredata = dict()
            featuredata["train"] = FSRawVectors.fmttrainingexamples 
            featuredata["test"] = FSRawVectors.fmttestexamples
            featuredata["labels"] = FSRawVectors.labels
            FSRawVectors.vectorlength = len(FSRawVectors.ingredients.keys())
            featuredata["vectorlength"] = FSRawVectors.vectorlength
            
            json.dump(featuredata,featurefile)            
            featurefile.flush()
            featurefile.close()
        else:
            featurefile = open("features/FSRawVectors.json")            
            featuredata = json.load(featurefile)
            FSRawVectors.fmttrainingexamples = featuredata["train"]
            FSRawVectors.fmttestexamples = featuredata["test"] 
            templabels = featuredata["labels"]
            FSRawVectors.vectorlength = featuredata["vectorlength"]
            for key in templabels.keys():
                FSRawVectors.labels[int(key)] = templabels[key]
            featurefile.close()        
       
    def formatexample(self,example):
        try:
            label = FSRawVectors.cuisines[example["cuisine"]]
        except KeyError:
            label = None
        try:
            id = FSRawVectors.cuisines[example["id"]]
        except KeyError:
            id = 0
        featurevector = list()
        for ingredient in example["ingredients"]:
            try:
                featurevector.append(FSRawVectors.ingredients[ingredient])
            except KeyError:
                pass   
        return [label,featurevector, id]
  
    def formattedtrainingexamples(self):
        return FSRawVectors.fmttrainingexamples 

    def formattedtestexamples(self):
        return FSRawVectors.fmttestexamples  
    
    def label(self, fmt_label):
        return FSRawVectors.labels[fmt_label]
      
    def labelfromexample(self,fmt_example):
        return self.label(fmt_example[0])

    def formattedlabels(self):
        return FSRawVectors.labels.keys()

    def formattedexamplevectorlength(self):
        return FSRawVectors.vectorlength

    def idfromexample(self,example):
        return example[2]
