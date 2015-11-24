import algorithm

class PerceptronBinaryClassifier:
    "Perceptron Binary Classifier"
    mytargetlabel = 0
    weights = None
    bias = 0.0
    cachedweights = None
    cachedbias = 0.0
    c = 1.0
    name = None    
    
    def __init__(self, targetlabel, weightvectorsize, name):
        self.mytargetlabel = targetlabel
        self.weights = [0.0]*weightvectorsize 
        self.cachedweights = [0.0]*weightvectorsize
        self.cachedbias = 0.0
        self.c = 1.0
        self.bias = 0.0
        self.name = name
    
    def trainclassifier(self, examplevectors, maxiterations):            
        iterations = 0
        for i in range(maxiterations):      
            iterations += 1
            exampleswithoutmistake = 0
            for example in examplevectors:
                sign = 1 if example[0] == self.mytargetlabel else -1
                dotproduct = 0.0
                for index in example[1]:
                    dotproduct += self.weights[index]
                if (sign*(dotproduct + self.bias) <= 0):
                    # mistake!
                    exampleswithoutmistake = 0
                    for index in example[1]:
                        self.weights[index] += sign
                        self.cachedweights[index] += (sign*self.c)
                    self.bias += sign
                    self.cachedbias += self.c*sign
                else:
                    exampleswithoutmistake += 1
                self.c += 1.0
            if (exampleswithoutmistake >= len(examplevectors)):
                break         
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - self.cachedweights[i]/self.c
        self.bias = self.bias - (self.cachedbias/self.c)    

    def predict(self, example):        
        dotproduct = 0.0
        for index in example[1]:
            dotproduct += self.weights[index]
        dotproduct += self.bias
        return dotproduct

    def name(self):
        return self.name

    def mytargetlabel(self):
        return mytargetlabel

class AlgPerceptronOVA(algorithm.Algorithm):
    "Implementation of Perceptron One vs. All Classifier with base class 'algorithm'"
    binaryclassifiers = list()
    cuisines = dict()
    labels = dict()    
    ingredients = dict()
    
          
    def extractfeatures(self,trainingexamples):
        ingredientindex = 0 
        cuisineindex = 0
        for example in trainingexamples:
            try: 
                cindex = AlgPerceptronOVA.cuisines[example["cuisine"]]                
            except KeyError:
                cindex = cuisineindex
                AlgPerceptronOVA.cuisines[example["cuisine"]] = cindex                
                AlgPerceptronOVA.labels[cindex] = example["cuisine"]
                cuisineindex += 1                            
            for ingredient in example["ingredients"]: 
                try: 
                    iindex = AlgPerceptronOVA.ingredients[ingredient]
                except KeyError:
                    iindex = ingredientindex
                    AlgPerceptronOVA.ingredients[ingredient] = iindex
                    ingredientindex += 1
        
    def formatexample(self,example):
        try:
            label = AlgPerceptronOVA.cuisines[example["cuisine"]]
        except KeyError:
            label = None
        featurevector = list()
        for ingredient in example["ingredients"]:
            try:
                featurevector.append(AlgPerceptronOVA.ingredients[ingredient])
            except KeyError:
                pass   
        return [label,featurevector]

    def train(self, trainingexamples):        
        ## create a classifier for each label
        for cuisine in AlgPerceptronOVA.cuisines.keys():
            AlgPerceptronOVA.binaryclassifiers.append(PerceptronBinaryClassifier(AlgPerceptronOVA.cuisines[cuisine], len(AlgPerceptronOVA.ingredients),cuisine))
        ## train each classifier on every example
        for classifier in AlgPerceptronOVA.binaryclassifiers:
            classifier.trainclassifier(trainingexamples,30)        

    def predict(self, example):
        results = list()        
        for classifier in AlgPerceptronOVA.binaryclassifiers:
            result = classifier.predict(example)
            results.append([result,classifier.mytargetlabel])
        sortedresults = sorted(results, cmp=resultcompare)
        return sortedresults[0][1]

    def name(self):
        return "Perceptron One vs. All"

    def label(self, fmt_label):
        return AlgPerceptronOVA.labels[fmt_label]

    def description(self):
        pass
        
    def labelfromexample(self,fmt_example):
        return self.label(fmt_example[0])


def resultcompare(x,y):
    return -1 if (x[0] - y[0]) > 0.0 else 1 