# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 09:21:18 2020

@author: kshit
"""

import nltk
import pandas as pd

documentA = 'Climate change is the  important crisis in 21st century day with drastic change in climate'
documentB = 'The impacts of climate change have been of the globe over the past years by increasing artic sea ice'
documentC = 'The artic is melting ice capads are melting at a very high rate causing rise in water level'
documentD ='ice capades water leve of climate elements arctic sea ice increased tendencies for drought and increase in the intensities of cyclonic storms'
documentE = 'It is projected that climate change by the end of the 21st century, many animal species will go extinct because of climate change'
documentF = 'The projected to increase at a high rate in all parts of world however reductions are also projected'
documentG ='climate extremes are projected to become more intense and frequent as a consequence of climate change For instance it is projected that the annual highest daily temperature increase'
documentH ='Similarly extreme precipitation amounts accumulated over a day or less are projected to increase in the future as a consequence of climate change'
documentI = 'As a result buildings and infrastructures across globe are expected to be exposed to annual and seasonal climate conditions'
documentJ = ' Failure to stop temperature increase by 21st century of climate could lead to the failure of world buildings and infrastructures'


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

bigrams = nltk.collocations.BigramAssocMeasures()
trigrams = nltk.collocations.TrigramAssocMeasures()
bigramFinderA = nltk.collocations.BigramCollocationFinder.from_words(filtered_sentenceA+filtered_sentenceB+filtered_sentenceC+filtered_sentenceD+filtered_sentenceE+filtered_sentenceF+filtered_sentenceG+filtered_sentenceH+filtered_sentenceI+filtered_sentenceJ)
#bigramFinderA.apply_freq_filter(2)

#trigramFinder.apply_freq_filter(20)
bigramPMITableA = pd.DataFrame(list(bigramFinderA.score_ngrams(bigrams.pmi)), columns=['bigram','PMI']).sort_values(by='PMI', ascending=False)

print(bigramPMITableA)
