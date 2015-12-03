import enchant
import operator
import nltk
import numpy

d = enchant.Dict("en_US")

filterlist = ['freshly','fresh','chopped','dried','crumbled','ground','shredded','red','green','black','blue',
              'orange','purple','yellow','powder','low','sodium','sauce','white','knower','brown','seasoning','juice','slices']

def generatefeaturestats(examples):    
    ingredientdict = dict()
    cuisines = dict()    
    for example in examples:
        try:
            cuisineingredients = cuisines[example["cuisine"]]
        except KeyError:
            cuisineingredients = dict()
            cuisines[example["cuisine"]] = cuisineingredients
        
        for ingredient in example["ingredients"]:
            try:
                cuisineingredients[ingredient] += 1
            except KeyError:
                cuisineingredients[ingredient] = 1
            try:
                ingredientdict[ingredient] += 1
            except KeyError:
                ingredientdict[ingredient] = 1

    print("Ingredients by frequency:")
    sorted_ingrededients = sorted(ingredientdict.items(), key=operator.itemgetter(1))
    for sortitem in sorted_ingrededients:
        print(sortitem)
    print("\r\n\r\n")

    for cuisine in cuisines.keys():
        print("Ingredients by frequency for cuisine: " + cuisine)
        cuisineingredients = cuisines[cuisine]
        sorted_ingrededients = sorted(cuisineingredients.items(), key=operator.itemgetter(1))
        for sortitem in sorted_ingrededients:
            print(sortitem)
        print("\r\n\r\n")

def cleanfeatureset(examples):
    cleanexamples = list()    
    for example in examples:
        cleanexample = dict()
        try:
            cleanexample["cuisine"] = example["cuisine"]       
        except:
            pass
        cleanexample["ingredients"] = list()
        for ingredient in example["ingredients"]:
            cleanexample["ingredients"].append(getCleanIngredient(ingredient));
        cleanexamples.append(cleanexample)                           
    return cleanexamples

def getCleanIngredient(phrase):
    tokens = phrase.lower().split(' ')
    cleantokens = ""
    for token in tokens:            
        if (token != "") and token not in filterlist:            
            if not d.check(token):
                try:
                    correcttoken = d.suggest(token)[0]
                    if (nltk.distance.edit_distance(correcttoken,token) <= 3):
                        cleantokens += (correcttoken + ' ')
                    else:
                        cleantokens += token
                except:
                    cleantokens += (token + ' ')
            else:
                cleantokens += (token + ' ')            
    return cleantokens.strip()

def getwordembeddingdictionaries(examples):
    wordtoindexembeddingdictionary = dict()        
    indextowordembeddingdictionary = dict()
    index = 0
    for example in examples:        
        for ingredient in example["ingredients"]:
            tokens = getCleanTokens(ingredient)
            for token in tokens:
                if token not in wordtoindexembeddingdictionary.keys():
                    wordtoindexembeddingdictionary[token] = index
                    indextowordembeddingdictionary[index] = token
                    index += 1                    
    return wordtoindexembeddingdictionary,indextowordembeddingdictionary

def getingredientlist(examples):
    ingredientdict = dict()
    for example in examples:
        for ingredient in example["ingredients"]:
            ingredientdict[ingredient] = 0
    return ingredientdict.keys()

def getingredientembeddedwordvector(dictionary, example):
    vector = [0.0]*len(dictionary.keys())
    tokens = getCleanTokens(example)
    for token in tokens:
        try:
            vector[dictionary[token]] += 1
        except KeyError:
            pass
    return vector

def getingredientvectors(dictionary, ingredients):   
    ingredientvectors = list()
    for ingredient in ingredients:
        ingredientvectors.append(getingredientembeddedwordvector(dictionary, ingredient))
    return ingredientvectors

def geteuclideaningredientclusterer(ingredientvectors, k, r):        
    clusterer = nltk.cluster.kmeans.KMeansClusterer(k,nltk.cluster.euclidean_distance, repeats = r, avoid_empty_clusters = True)
    vectors = [numpy.array(vector) for vector in ingredientvectors]
    clusters = clusterer.cluster(vectors, True)    
    return clusterer

def getcosineingredientclusterer(ingredientvectors, k, r):        
    clusterer = nltk.cluster.kmeans.KMeansClusterer(k,nltk.cluster.cosine_distance, repeats = r, avoid_empty_clusters = True)
    vectors = [numpy.array(vector) for vector in ingredientvectors]
    clusters = clusterer.cluster(vectors, True)    
    return clusterer

def testclusterer(clusterer, ingredientsvectors, indextoworddictionary):
    clusterdict = dict()
    for vector in ingredientsvectors:
        cluster = clusterer.classify(vector)
        try:
            clusterdict[cluster].append(vector)
        except KeyError:
            clusterdict[cluster] = [vector]

    for key in clusterdict.keys():        
        print ("Begin Cluster")
        for vector in clusterdict[key]:
            output = ""            
            for i in range(len(vector)):
                if vector[i] != 0:
                    output += indextoworddictionary[i]
            print(output.encode('ascii', 'ignore'))
        print ("End Cluster")

translation_table = dict.fromkeys(map(ord, '.,!?[]/*\'$&\"'), None)
                    
def getCleanTokens(sentence):
    tokens = sentence.split(' ')
    cleantokens = list()
    for token in tokens:
        #cleantoken = token.translate(None, '')
        if (token != ""):
            cleantokens.append(token)
    return cleantokens

def runclusterroutine(examples):
    ingredientlist = getingredientlist(examples)
    word2index,index2word = getwordembeddingdictionaries(examples)
    vectors = getingredientvectors(word2index, ingredientlist)
    clusterer = geteuclideaningredientclusterer(vectors)
    testclusterer(clusterer, vectors, index2word)