# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 16:22:28 2020

@author: kshit
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

documentA = 'Climate change is extreme crisis today with drastic change in climate'
documentB = 'The extreme impact of climate change have been detected  on the globe the years'
documentC = 'The artic is melting ice capades are melting at a  increase rate bad climate'
documentD ='drastic changes in extreme of climate such as decreases in Arctic sea ice'
documentE = ' It is projected that drastic temeprature difference and climate change'
documentF = 'The climate winter precipitation is projected to increase but decrease in summer precipitation are also projected'
documentG ='climate extremes are projected to become more intense as a consequence of climate projected that increase daily temperature'
documentH =' extreme precipitation  are projected to increase as a consequence of climate change'
documentI = 'buildings across globe are expected to be exposed to drastic different climate conditions and extreme events'
documentJ = ' Failure for the movies of climate to the movies of movies'


bagOfWordsA = documentA.lower().split(' ')
bagOfWordsB = documentB.lower().split(' ')
bagOfWordsC = documentC.lower().split(' ')
bagOfWordsD = documentD.lower().split(' ')
bagOfWordsE = documentE.lower().split(' ')
bagOfWordsF = documentF.lower().split(' ')
bagOfWordsG = documentG.lower().split(' ')
bagOfWordsH = documentH.lower().split(' ')
bagOfWordsI = documentI.lower().split(' ')
bagOfWordsJ = documentJ.lower().split(' ')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')) 

uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB)).union(set(bagOfWordsC)).union(set(bagOfWordsD)).union(set(bagOfWordsE)).union(set(bagOfWordsF)).union(set(bagOfWordsG)).union(set(bagOfWordsH)).union(set(bagOfWordsI)).union(set(bagOfWordsJ))

uniqueWords = [w for w in uniqueWords if not w in stop_words]

filtered_sentenceA = [w for w in bagOfWordsA if not w in stop_words] 

filtered_sentenceB = [w for w in bagOfWordsB if not w in stop_words] 

filtered_sentenceC = [w for w in bagOfWordsC if not w in stop_words] 

filtered_sentenceD = [w for w in bagOfWordsD if not w in stop_words] 

filtered_sentenceE = [w for w in bagOfWordsE if not w in stop_words] 

filtered_sentenceF = [w for w in bagOfWordsF if not w in stop_words] 

filtered_sentenceG = [w for w in bagOfWordsG if not w in stop_words] 

filtered_sentenceH = [w for w in bagOfWordsH if not w in stop_words] 

filtered_sentenceI = [w for w in bagOfWordsI if not w in stop_words] 

filtered_sentenceJ = [w for w in bagOfWordsJ if not w in stop_words]
numOfWordsA = dict.fromkeys(uniqueWords, 0)
for word in filtered_sentenceA:
    numOfWordsA[word] += 1
numOfWordsB = dict.fromkeys(uniqueWords, 0)
for word in filtered_sentenceB:
    numOfWordsB[word] += 1
    
numOfWordsC = dict.fromkeys(uniqueWords, 0)
for word in filtered_sentenceC:
    numOfWordsC[word] += 1
numOfWordsD = dict.fromkeys(uniqueWords, 0)
for word in filtered_sentenceD:
    numOfWordsD[word] += 1

numOfWordsE = dict.fromkeys(uniqueWords, 0)
for word in filtered_sentenceE:
    numOfWordsE[word] += 1
numOfWordsF = dict.fromkeys(uniqueWords, 0)
for word in filtered_sentenceF:
    numOfWordsF[word] += 1
    
numOfWordsG = dict.fromkeys(uniqueWords, 0)
for word in filtered_sentenceG:
    numOfWordsG[word] += 1
numOfWordsH = dict.fromkeys(uniqueWords, 0)
for word in filtered_sentenceH:
    numOfWordsH[word] += 1
    
numOfWordsI = dict.fromkeys(uniqueWords, 0)
for word in filtered_sentenceI:
    numOfWordsI[word] += 1
numOfWordsJ = dict.fromkeys(uniqueWords, 0)
for word in filtered_sentenceJ:
    numOfWordsJ[word] += 1
    

def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float (bagOfWordsCount)
    return tfDict

tfA = computeTF(numOfWordsA, filtered_sentenceA)
tfB = computeTF(numOfWordsB, filtered_sentenceB)
tfC = computeTF(numOfWordsC, filtered_sentenceC)
tfD = computeTF(numOfWordsD, filtered_sentenceD)
tfE = computeTF(numOfWordsE, filtered_sentenceE)
tfF = computeTF(numOfWordsF, filtered_sentenceF)
tfG = computeTF(numOfWordsG, filtered_sentenceG)
tfH = computeTF(numOfWordsH, filtered_sentenceH)
tfI = computeTF(numOfWordsI, filtered_sentenceI)
tfJ = computeTF(numOfWordsJ, filtered_sentenceJ)

print(tfA)
print("\n")
print(tfB)
print("\n")
print(tfC)
print("\n")
print(tfD)
print("\n")
print(tfE)
print("\n")
print(tfF)
print("\n")
print(tfG)
print("\n")
print(tfH)
print("\n")
print(tfI)
print("\n")
print(tfJ)

df1= pd.DataFrame([tfA, tfB,tfC, tfD,tfE, tfF,tfG, tfH,tfI, tfJ])
print(df1)
from wordcloud import WordCloud
import matplotlib.pyplot as plt

comment_words =str(bagOfWordsJ)

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords = stop_words, 
                min_font_size = 8).generate_from_frequencies(tfA)
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
wordcloud2 = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords = stop_words, 
                min_font_size = 8).generate_from_frequencies(tfB)
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud2) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

wordcloud3 = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords = stop_words, 
                min_font_size = 8).generate_from_frequencies(tfC)
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud3) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

wordcloud4 = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords = stop_words, 
                min_font_size = 8).generate_from_frequencies(tfD)
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud4) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

wordcloud5 = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords = stop_words, 
                min_font_size = 8).generate_from_frequencies(tfE)
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud5) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

wordcloud6 = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords = stop_words, 
                min_font_size = 8).generate_from_frequencies(tfF)
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud6) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

wordcloud7 = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords = stop_words, 
                min_font_size = 8).generate_from_frequencies(tfG)
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud7) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

wordcloud8 = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords = stop_words, 
                min_font_size = 8).generate_from_frequencies(tfH)
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud8) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

wordcloud9 = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords = stop_words, 
                min_font_size = 8).generate_from_frequencies(tfI)
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud9) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

wordcloud10 = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords = stop_words, 
                min_font_size = 8).generate_from_frequencies(tfJ)
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud10) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
def computeIDF(documents):
    import math
    N = len(documents)
    
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

idfs = computeIDF([numOfWordsA, numOfWordsB,numOfWordsC, numOfWordsD, numOfWordsE, numOfWordsF, numOfWordsG, numOfWordsH, numOfWordsI, numOfWordsJ])

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)
tfidfC = computeTFIDF(tfC, idfs)
tfidfD = computeTFIDF(tfD, idfs)
tfidfE = computeTFIDF(tfE, idfs)
tfidfF = computeTFIDF(tfF, idfs)
tfidfG = computeTFIDF(tfG, idfs)
tfidfH = computeTFIDF(tfH, idfs)
tfidfI = computeTFIDF(tfI, idfs)
tfidfJ = computeTFIDF(tfJ, idfs)
df = pd.DataFrame([tfidfA, tfidfB,tfidfC, tfidfD,tfidfE, tfidfF,tfidfG, tfidfH,tfidfI, tfidfJ]) #tfidf

#https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76