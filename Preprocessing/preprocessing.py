# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 19:15:59 2020

@author: kshit
"""

import nltk
from nltk.corpus import stopwords
file = open('test.txt','r')
data = file.read()

lines= data.split(".")
#data=data.lower()

print("\nContents of the file to be analyzed is as follows\n",data)
data_token=nltk.word_tokenize(data)

print("\nTokenized data\n",data_token)

#nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english')) 
  
  
filtered_sentence = [w for w in data_token if not w in stop_words] 

print("\nData after removing stop words\n",filtered_sentence)


pst=nltk.pos_tag(data_token)
print(pst)


from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
wnl = WordNetLemmatizer()

pst=nltk.pos_tag(data_token)
print(pst)
for word, tag in pst:
     wntag = tag[0].lower()
     wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
     if not wntag:
             lemma = word
     else:
             lemmaWord = wnl.lemmatize(word, wntag)
             print("Lemmatized data",lemmaWord)

      

porter_stem = PorterStemmer()

stem_data = []
for word in data_token:
    stem_data.append(porter_stem.stem(word))
    
print("\nData after performing Porter Stemming\n", stem_data)

'''
from nltk.tokenize import sent_tokenize
sent_token = []

for line in lines:
    sent_token.append(sent_tokenize(line))
    '''