import algorithm

class PerceptronBinaryClassifier:
    "Perceptron Binary Classifier"
    mytargetlabel = 0
    weights = None
    bias = 0.0
    cachedweights = [0.0]*len(ngrams)
    cachedbias = 0.0
    c = 1.0
    name = None    
    
    def __init__(self, targetlabel, weightvectorsize, name):
        self.mytargetlabel = targetlabel
        self.weights = [0.0]*weightvectorsize 
        self.name = name
    
    def trainclassifier(self, examplevectors, maxiterations):            
        iterations = 0
        for i in range(maxiterations):      
            iterations += 1
            exampleswithoutmistake = 0
            for example in examplevectors:
                sign = 1 if example[0] == self.mytargetlabel else -1
                dotproduct = 0.0
                for j in range(1,len(example),1):
                    dotproduct += weights[example[j]]
                if (sign*(dotproduct + bias) <= 0):
                    # mistake!
                    exampleswithoutmistake = 0
                    for k in range(1,len(example),1):
                        self.weights[example[k]] += sign
                        self.cachedweights[example[k]] += (sign*c)
                    self.bias += sign
                    self.cachedbias += c*sign
                else:
                    exampleswithoutmistake += 1
                self.c += 1.0
            if (exampleswithoutmistake >= len(examplevectors)):
                break         
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - self.cachedweights[i]/c
        self.bias = self.bias - (self.cachedbias/self.c)    

    def predict(self, example):
        euclidiandistance = 0.0
        dotproduct = 0.0
        for i in range(1,len(example),1):
            dotproduct += weights[example[i]]
        dotproduct += bias
        return [-1, dotproduct] if dotproduct <= threshold else [1,dotproduct]

    def name(self):
        return self.name

class AlgPerceptronOVA(algorithm.Algorithm):
    "Implementation of Perceptron One vs. All Classifier with base class 'algorithm'"
    binaryclassifiers = dict()

    def train(self, trainingexamples):
        ## create a set of all of the labels
        ## create a set of all of the ingredients
        cuisines = dict()
        cuisineindex = 0
        ingredients = dict()
        ingredientindex = 0
        examplevectors = list()
        for example in trainingexamples:            
            examplevector = list()
            try: 
                cindex = cuisines[example["cuisine"]]
            except KeyError:
                cindex = cuisineindex
                cuisines[example["cuisine"]] = cindex                
                cuisineIndex += 1
            examplevector.append(cindex)                
            for ingredient in example["ingredients"]: 
                try: 
                    iindex = ingredients[ingredient]
                except KeyError:
                    iindex = ingredientindex
                    ingredients[ingredient] = iindex
                    ingredientindex += 1
            examplevectors.append(examplevector)
        ## create a classifier for each label
        for cuisine in cuisines.keys():
            binaryclassifiers.append(PerceptronBinaryClassifier(cuisines[cuisine], len(ingredients)))
        ## train each classifier on every example
        for classifier in binaryclassifiers:
            classifier.train(examplevectors)        

    def predict(self, example):
        results = list()        
        for classifier in binaryclassifiers:
            result,distance = classifier.predict(example)
            results.append([result,distance,classifier])
        results = results.sort(resultcompare)
        return result[0][2].name()

    def resultcompare(x,y):
        if (x[0] == 1 and y[0] == 1):
            return x[1] - y[1]
        elif (x[0] == 1 and y[0] == -1):
            return 1
        elif (x[0] == -1 and y[0] == 1):
            return -1
        else:
            return y[1] - x[1]
                

    def name(self):
        return "Perceptron One vs. All"
