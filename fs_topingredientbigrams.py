import featureset
import os
import json

class FSTopIngredientBigramVectors(featureset.FeatureSet):
    """description of class"""
    cuisines = dict()
    labels = dict()    
    ingredients = dict()
    ingredienttupledict = dict()
    fmttrainingexamples = list()
    fmttestexamples = list()
    vectorlength = 0
    
    def extractfeatures(self,trainingexamples, testexamples):
        if not os.path.exists("features/FSTopIngredientBigramVectors.json"):
            featurefile = open("features/FSTopIngredientBigramVectors.json",'wb')            
            ingredientindex = 0 
            cuisineindex = 0
            jcstrainingvectors = list()
            for example in trainingexamples:
                formattedexample = list()
                fmtcuisine = None;
                try: 
                    cindex = FSTopIngredientBigramVectors.cuisines[example["cuisine"]]
                    fmtcuisine = cindex
                except KeyError:
                    cindex = cuisineindex
                    FSTopIngredientBigramVectors.cuisines[example["cuisine"]] = cindex                
                    FSTopIngredientBigramVectors.labels[cindex] = example["cuisine"]
                    cuisineindex += 1
                    fmtcuisine = cindex
                fmtingredients = list()                
                for ingredient in example["ingredients"]: 
                    try: 
                        iindex = FSTopIngredientBigramVectors.ingredients[ingredient]
                        fmtingredients.append(iindex)
                    except KeyError:
                        iindex = ingredientindex
                        FSTopIngredientBigramVectors.ingredients[ingredient] = iindex
                        ingredientindex += 1
                        fmtingredients.append(iindex)
                try:
                    id = FSTopIngredientBigramVectors.cuisines[example["id"]]
                except KeyError:
                    id = 0
                jcstrainingvectors.append([fmtcuisine,fmtingredients,id])

            jcstestvectors = list()
            for example in testexamples:
                formattedexample = list()
                fmtcuisine = None
                fmtingredients = list()                
                for ingredient in example["ingredients"]: 
                    try:
                        iindex = FSTopIngredientBigramVectors.ingredients[ingredient]
                        fmtingredients.append(iindex)                    
                    except KeyError:
                        pass
                try:
                    id = FSTopIngredientBigramVectors.cuisines[example["id"]]
                except KeyError:
                    id = 0
                jcstestvectors.append([fmtcuisine,fmtingredients,id])

            #compute frequency statistics for each cuisine
            cuisinetoingredientdict = dict()
            for vector in jcstrainingvectors:
                ingredientstats = None
                try: 
                    ingredientstats = cuisinetoingredientdict[vector[0]]                    
                except KeyError:
                    ingredientstats = dict()
                    cuisinetoingredientdict[vector[0]] = ingredientstats
                for iindex in vector[1]:
                    try:
                        ingredientfreq = ingredientstats[iindex]
                    except:
                        ingredientfreq = 0
                    ingredientfreq += 1    
                    ingredientstats[iindex] = ingredientfreq

            #get a sorted list of ingredients from each cuisine sorted by frequency
            cuisinetoingredientlists = dict()
            
            tindex = 0
            for key in cuisinetoingredientdict.keys():
                ingredientstats = cuisinetoingredientdict[key]
                ingredientfreqtuples = ingredientstats.items()
                ingredientfreqtuples = sorted(ingredientfreqtuples,key=lambda x: x[1])
                topten = ingredientfreqtuples[-20:]
                for ing1 in topten:
                    for ing2 in topten:
                        if (ing1[0] != ing2[0]):
                            mytuple = (ing1[0],ing2[0])
                            try:
                                tupleindex = FSTopIngredientBigramVectors.ingredienttupledict[mytuple]
                            except KeyError:
                                FSTopIngredientBigramVectors.ingredienttupledict[mytuple] = tindex
                                tindex += 1

            #Now we have a list of tuples to build the feature set            
            for vector in jcstrainingvectors:
                tuplevector = list()
                for ingtuple in FSTopIngredientBigramVectors.ingredienttupledict.keys():
                    if (ingtuple[0] in vector[1]) and (ingtuple[1] in vector[1]):
                        tuplevector.append(FSTopIngredientBigramVectors.ingredienttupledict[ingtuple])
                FSTopIngredientBigramVectors.fmttrainingexamples.append([vector[0],tuplevector, vector[2]])

            for vector in jcstestvectors:
                tuplevector = list()
                for ingtuple in FSTopIngredientBigramVectors.ingredienttupledict.keys():
                    if (ingtuple[0] in vector[1]) and (ingtuple[1] in vector[1]):
                        tuplevector.append(FSTopIngredientBigramVectors.ingredienttupledict[ingtuple])
                FSTopIngredientBigramVectors.fmttestexamples.append([vector[0],tuplevector, vector[2]])            
            
            FSTopIngredientBigramVectors.vectorlength = len(FSTopIngredientBigramVectors.ingredienttupledict.keys())
                 
            featuredata = dict()
            featuredata["train"] = FSTopIngredientBigramVectors.fmttrainingexamples 
            featuredata["test"] = FSTopIngredientBigramVectors.fmttestexamples
            featuredata["labels"] = FSTopIngredientBigramVectors.labels
            featuredata["vectorlength"] = FSTopIngredientBigramVectors.vectorlength
                                                      
            json.dump(featuredata,featurefile)            
            featurefile.flush()
            featurefile.close()
        else:
            featurefile = open("features/FSTopIngredientBigramVectors.json")            
            featuredata = json.load(featurefile)
            FSTopIngredientBigramVectors.fmttrainingexamples = featuredata["train"]
            FSTopIngredientBigramVectors.fmttestexamples = featuredata["test"] 
            templabels = featuredata["labels"]
            for key in templabels.keys():
                FSTopIngredientBigramVectors.labels[int(key)] = templabels[key]
            FSTopIngredientBigramVectors.vectorlength = featuredata["vectorlength"]
            featurefile.close()        
       
    def formatexample(self,example):
        try:
            label = FSTopIngredientBigramVectors.cuisines[example["cuisine"]]
        except KeyError:
            label = None
        
        featurevector = list()
        for ingredient in example["ingredients"]:
            try:
                featurevector.append(FSTopIngredientBigramVectors.ingredients[ingredient])
            except KeyError:
                pass   
        return [label,featurevector, id]
  
    def formattedtrainingexamples(self):
        return FSTopIngredientBigramVectors.fmttrainingexamples 

    def formattedtestexamples(self):
        return FSTopIngredientBigramVectors.fmttestexamples  
    
    def label(self, fmt_label):
        return FSTopIngredientBigramVectors.labels[fmt_label]
      
    def labelfromexample(self,fmt_example):
        return self.label(fmt_example[0])

    def formattedlabels(self):
        return FSTopIngredientBigramVectors.labels.keys()

    def formattedexamplevectorlength(self):
        return FSTopIngredientBigramVectors.vectorlength

    def idfromexample(self,example):
        return example[2]
