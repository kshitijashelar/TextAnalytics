# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 12:45:57 2020

@author: kshit
"""

# bits from http://stackoverflow.com/questions/15173225/how-to-calculate-cosine-similarity-given-2-sentence-strings-python
# load_docs, process_docs and compute_vector by MK
import math
from collections import Counter
from nltk.corpus import stopwords

vector_dict = {}

#Just loads in all the documents
def load_docs():
 print("Loading docs...")
 doc1=('d1', 'The September 11 attacks, often referred to as 9/11, were a series of four coordinated terrorist attacks by the Islamist terrorist group Al-Qaeda against the United States on the morning of Tuesday, September 11, 2001.Four passenger airliners which had departed from airports in the northeastern United States bound for California were hijacked by 19 al-Qaeda terrorists. Two of the planes, American Airlines Flight 11 and United Airlines Flight 175, crashed into the North and South towers, respectively, of the World Trade Center complex in Lower Manhattan. Almost 3,000 people were killed during the 9/11 terrorist attacks')
 doc2=('d2', 'On September 11, 2001, 19 militants associated with the Islamic extremist group al Qaeda hijacked four airplanes and carried out suicide attacks against targets in the United States. were a series of four coordinated terrorist attacks by the Islamist terrorist group Al-Qaeda against the United States on the morning of Tuesday, September 11, 2001. Two of the planes were flown into the twin towers of the World Trade Center in New York City, a third plane hit the Pentagon just outside Washington, D.C., and the fourth plane crashed in a field in Shanksville, Pennsylvania. Almost 3,000 people were killed during the 9/11 terrorist attacks, which triggered major U.S. initiatives to combat terrorism and defined the presidency of George W. Bush.')
 doc3=('d3', 'At the World Trade Center (WTC) site in Lower Manhattan, 2,753 people were killed when hijacked American Airlines Flight 11 and United Airlines Flight 175 were intentionally crashed into the north and south towers, or as a result of the crashes.were a series of four coordinated terrorist attacks by the Islamist terrorist group Al-Qaeda against the United States on the morning of Tuesday, September 11, 2001. Of those who perished during the initial attacks and the subsequent collapses of the towers, 343 were New York City firefighters, 23 were New York City police officers and 37 were officers at the Port Authority. Almost 3,000 people were killed during the 9/11 terrorist attacks. At the Pentagon in Washington, 184 people were killed when hijacked American Airlines Flight 77 crashed into the building. ')

 doc4=('d4', 'The 2008 Mumbai attacks  were series of terrorist attacks when 10 members of Lashkar-e-Taiba attacked Mumbai. carried out by 10 gunmen who were believed to be connected to Lashkar-e-Taiba, a Pakistan-based terrorist organization. Armed with automatic weapons and hand grenades, the terrorists targeted civilians at numerous sites in the southern part of Mumbai, including the Chhatrapati Shivaji Terminus railway station, the popular Leopold Café, two  Cama hospital, and a theatre Mumbai Chabad House. While most of the attacks ended within a few hours after they began at around 9:30 PM on November 26, the terror continued to unfold at three locations where hostages were taken—the Nariman House, where a Jewish outreach centre was located, and the luxury hotels Oberoi Trident and Taj Mahal Palace & Tower.')
 doc5=('d5', 'The 2008 Mumbai attacks (also referred to as 26/11) were a series of terrorist attacks that took place in November 2008, when 10 members of Lashkar-e-Taiba, an extremist Islamist terrorist organisation based in Pakistan, carried out 12 coordinated shooting and bombing attacks lasting four days across Mumbai. The attacks, which drew widespread global condemnation, began on Wednesday 26 November and lasted until Saturday 29 November 2008. At least 174 people died, including 9 attackers, and more than 300 were wounded. Eight of the attacks occurred in South Mumbai at Chhatrapati Shivaji Terminus, Mumbai Chabad House, The Oberoi Trident, The Taj Palace & Tower, Leopold Cafe, Cama Hospital, The Nariman House, the Metro Cinema, and in a lane behind the Times of India building and St. Xaviers College. ')
 doc6=('d6', 'The world remembers 26/11. The 2008 Mumbai attacks  were series of terrorist attack  when 10 members of Lashkar-e-Taiba attacked Mumbai. The victims lost their lives in the terror attacks at various places- Chhatrapati Shivaji Terminus, Mumbai Chabad House, The Oberoi Trident, The Taj Palace & Tower, Leopold Cafe, Cama Hospital, The Nariman House,the Metro Cinema, and in a lane behind the Times of India building and St. Xaviers College, eleven years ago which began on November 26, 2008.The attacks lasted for four days killing 166 people and injuring over 300. The Jamaat-ud-Dawah (JuD), whose mastermind was Hafiz Saeed, was believed to have plotted the 26/11 attacks.')

 return [doc1, doc2,doc3,doc4,doc5,doc6]

#Computes TF for words in each doc, DF for all features in all docs; finally whole Tf-IDF matrix
def process_docs(all_dcs):
 stop_words = set(stopwords.words('english')) 
 all_words = []
 counts_dict = {}
 for doc in all_dcs:
    words = [x.lower() for x in doc[1].split() if x not in stop_words]
    words_counted = Counter(words)
    unique_words = list(words_counted.keys())
    counts_dict[doc[0]] = words_counted
    all_words = all_words + unique_words
 n = len(counts_dict)
 df_counts = Counter(all_words)
 compute_vector_len(counts_dict, n, df_counts)


#computes TF-IDF for all words in all docs
def compute_vector_len(doc_dict, no, df_counts):
  global vector_dict
  for doc_name in doc_dict:
    doc_words = doc_dict[doc_name].keys()
    wd_tfidf_scores = {}
    for wd in list(set(doc_words)):
        wds_cts = doc_dict[doc_name]
        wd_tf_idf = wds_cts[wd] * math.log(no / df_counts[wd], 10)
        wd_tfidf_scores[wd] = round(wd_tf_idf, 4)
    vector_dict[doc_name] = wd_tfidf_scores

def get_cosine(text1, text2):
     vec1 = vector_dict[text1]
     vec2 = vector_dict[text2]
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])
     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)
     if not denominator:
        return 0.0
     else:
        return round(float(numerator) / denominator, 3)

#RUN
all_docs = load_docs()
process_docs(all_docs)
text = ['d1','d2','d3','d4','d5','d6']
ans = [[999,999,999,999,999,999],[999,999,999,999,999,999],[999,999,999,999,999,999],[999,999,999,999,999,999],[999,999,999,999,999,999],[999,999,999,999,999,999]]

for i in range(0,6):
    for j in range(0,6):
        ans[i][j] = get_cosine(text[i], text[j])
        
        

import matplotlib.pyplot as plt 
import numpy as np
import sklearn
#%matplotlib inline 

fig, ax = plt.subplots()
width = 0.35
'''
# Add the prior figures to the data for plotting
objects =  list(text)
positive =  list(ans)
#res = list(bias_d1.values())

y_pos = np.arange(len(objects))

p1 = ax.bar(y_pos, positive, width, align='center', 
            color=[ 'pink', 'pink','pink','pink'],alpha=0.5)

#p2 = ax.bar(y_pos+width, res, width, align='center', color=[ 'maroon','maroon','maroon','maroon'],alpha=0.5)

#ax.legend((p1[1], p2[1]), ('Raw', 'Upsampled'))

plt.xticks(y_pos, objects)
plt.ylabel('Minority Counts')
plt.title('Impact of Upsampling ADASYN')
 
plt.show()

# Display matrix'''
plt.matshow(ans)

plt.show()
'''
doc1=["The September 11 attacks, often referred to as 9/11, were a series of four coordinated terrorist attacks by the Islamist terrorist group Al-Qaeda against the United States on the morning of Tuesday, September 11, 2001.Four passenger airliners which had departed from airports in the northeastern United States bound for California were hijacked by 19 al-Qaeda terrorists. Two of the planes, American Airlines Flight 11 and United Airlines Flight 175, crashed into the North and South towers, respectively, of the World Trade Center complex in Lower Manhattan. Almost 3,000 people were killed during the 9/11 terrorist attacks"]
doc2= ["On September 11, 2001, 19 militants associated with the Islamic extremist group al Qaeda hijacked four airplanes and carried out suicide attacks against targets in the United States. were a series of four coordinated terrorist attacks by the Islamist terrorist group Al-Qaeda against the United States on the morning of Tuesday, September 11, 2001. Two of the planes were flown into the twin towers of the World Trade Center in New York City, a third plane hit the Pentagon just outside Washington, D.C., and the fourth plane crashed in a field in Shanksville, Pennsylvania. Almost 3,000 people were killed during the 9/11 terrorist attacks, which triggered major U.S. initiatives to combat terrorism and defined the presidency of George W. Bush."]

print(sklearn.metrics.pairwise.cosine_similarity(text[1], text[2], dense_output=True))
'''


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import *
from sklearn.feature_extraction.text import TfidfVectorizer

text1= ["The September 11 attacks, often referred to as 9/11, were a series of four coordinated terrorist attacks by the Islamist terrorist group Al-Qaeda against the United States on the morning of Tuesday, September 11, 2001.Four passenger airliners which had departed from airports in the northeastern United States bound for California were hijacked by 19 al-Qaeda terrorists. Two of the planes, American Airlines Flight 11 and United Airlines Flight 175, crashed into the North and South towers, respectively, of the World Trade Center complex in Lower Manhattan. Almost 3,000 people were killed during the 9/11 terrorist attacks", 
        "On September 11, 2001, 19 militants associated with the Islamic extremist group al Qaeda hijacked four airplanes and carried out suicide attacks against targets in the United States. were a series of four coordinated terrorist attacks by the Islamist terrorist group Al-Qaeda against the United States on the morning of Tuesday, September 11, 2001. Two of the planes were flown into the twin towers of the World Trade Center in New York City, a third plane hit the Pentagon just outside Washington, D.C., and the fourth plane crashed in a field in Shanksville, Pennsylvania. Almost 3,000 people were killed during the 9/11 terrorist attacks, which triggered major U.S. initiatives to combat terrorism and defined the presidency of George W. Bush.",
        "At the World Trade Center (WTC) site in Lower Manhattan, 2,753 people were killed when hijacked American Airlines Flight 11 and United Airlines Flight 175 were intentionally crashed into the north and south towers, or as a result of the crashes.were a series of four coordinated terrorist attacks by the Islamist terrorist group Al-Qaeda against the United States on the morning of Tuesday, September 11, 2001. Of those who perished during the initial attacks and the subsequent collapses of the towers, 343 were New York City firefighters, 23 were New York City police officers and 37 were officers at the Port Authority. Almost 3,000 people were killed during the 9/11 terrorist attacks. At the Pentagon in Washington, 184 people were killed when hijacked American Airlines Flight 77 crashed into the building.",
        "The 2008 Mumbai attacks  were series of terrorist attacks when 10 members of Lashkar-e-Taiba attacked Mumbai. carried out by 10 gunmen who were believed to be connected to Lashkar-e-Taiba, a Pakistan-based terrorist organization. Armed with automatic weapons and hand grenades, the terrorists targeted civilians at numerous sites in the southern part of Mumbai, including the Chhatrapati Shivaji Terminus railway station, the popular Leopold Café, two  Cama hospital, and a theatre Mumbai Chabad House. While most of the attacks ended within a few hours after they began at around 9:30 PM on November 26, the terror continued to unfold at three locations where hostages were taken—the Nariman House, where a Jewish outreach centre was located, and the luxury hotels Oberoi Trident and Taj Mahal Palace & Tower.",
        "The 2008 Mumbai attacks (also referred to as 26/11) were a series of terrorist attacks that took place in November 2008, when 10 members of Lashkar-e-Taiba, an extremist Islamist terrorist organisation based in Pakistan, carried out 12 coordinated shooting and bombing attacks lasting four days across Mumbai. The attacks, which drew widespread global condemnation, began on Wednesday 26 November and lasted until Saturday 29 November 2008. At least 174 people died, including 9 attackers, and more than 300 were wounded. Eight of the attacks occurred in South Mumbai at Chhatrapati Shivaji Terminus, Mumbai Chabad House, The Oberoi Trident, The Taj Palace & Tower, Leopold Cafe, Cama Hospital, The Nariman House, the Metro Cinema, and in a lane behind the Times of India building and St. Xaviers College.",
        "The world remembers 26/11. The 2008 Mumbai attacks  were series of terrorist attack  when 10 members of Lashkar-e-Taiba attacked Mumbai. The victims lost their lives in the terror attacks at various places- Chhatrapati Shivaji Terminus, Mumbai Chabad House, The Oberoi Trident, The Taj Palace & Tower, Leopold Cafe, Cama Hospital, The Nariman House,the Metro Cinema, and in a lane behind the Times of India building and St. Xaviers College, eleven years ago which began on November 26, 2008.The attacks lasted for four days killing 166 people and injuring over 300. The Jamaat-ud-Dawah (JuD), whose mastermind was Hafiz Saeed, was believed to have plotted the 26/11 attacks."]


'''text2=text2.lower()
text1= text1.lower()
text2=
doc1=('d1', text1)

doc2=('d2',text2)'''
'''
text3 =
doc3=('d3', 'At the World Trade Center (WTC) site in Lower Manhattan, 2,753 people were killed when hijacked American Airlines Flight 11 and United Airlines Flight 175 were intentionally crashed into the north and south towers, or as a result of the crashes.were a series of four coordinated terrorist attacks by the Islamist terrorist group Al-Qaeda against the United States on the morning of Tuesday, September 11, 2001. Of those who perished during the initial attacks and the subsequent collapses of the towers, 343 were New York City firefighters, 23 were New York City police officers and 37 were officers at the Port Authority. Almost 3,000 people were killed during the 9/11 terrorist attacks. At the Pentagon in Washington, 184 people were killed when hijacked American Airlines Flight 77 crashed into the building. ')

doc4=('d4', 'The 2008 Mumbai attacks  were series of terrorist attacks when 10 members of Lashkar-e-Taiba attacked Mumbai. carried out by 10 gunmen who were believed to be connected to Lashkar-e-Taiba, a Pakistan-based terrorist organization. Armed with automatic weapons and hand grenades, the terrorists targeted civilians at numerous sites in the southern part of Mumbai, including the Chhatrapati Shivaji Terminus railway station, the popular Leopold Café, two  Cama hospital, and a theatre Mumbai Chabad House. While most of the attacks ended within a few hours after they began at around 9:30 PM on November 26, the terror continued to unfold at three locations where hostages were taken—the Nariman House, where a Jewish outreach centre was located, and the luxury hotels Oberoi Trident and Taj Mahal Palace & Tower.')
doc5=('d5', 'The 2008 Mumbai attacks (also referred to as 26/11) were a series of terrorist attacks that took place in November 2008, when 10 members of Lashkar-e-Taiba, an extremist Islamist terrorist organisation based in Pakistan, carried out 12 coordinated shooting and bombing attacks lasting four days across Mumbai. The attacks, which drew widespread global condemnation, began on Wednesday 26 November and lasted until Saturday 29 November 2008. At least 174 people died, including 9 attackers, and more than 300 were wounded. Eight of the attacks occurred in South Mumbai at Chhatrapati Shivaji Terminus, Mumbai Chabad House, The Oberoi Trident, The Taj Palace & Tower, Leopold Cafe, Cama Hospital, The Nariman House, the Metro Cinema, and in a lane behind the Times of India building and St. Xaviers College. ')
doc6=('d6', 'The world remembers 26/11. The 2008 Mumbai attacks  were series of terrorist attack  when 10 members of Lashkar-e-Taiba attacked Mumbai. The victims lost their lives in the terror attacks at various places- Chhatrapati Shivaji Terminus, Mumbai Chabad House, The Oberoi Trident, The Taj Palace & Tower, Leopold Cafe, Cama Hospital, The Nariman House,the Metro Cinema, and in a lane behind the Times of India building and St. Xaviers College, eleven years ago which began on November 26, 2008.The attacks lasted for four days killing 166 people and injuring over 300. The Jamaat-ud-Dawah (JuD), whose mastermind was Hafiz Saeed, was believed to have plotted the 26/11 attacks.')'''
#train_set = [doc1, doc2]
#ans1 = [[999,999,999,999,999,999],[999,999,999,999,999,999],[999,999,999,999,999,999],[999,999,999,999,999,999],[999,999,999,999,999,999],[999,999,999,999,999,999]]

tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
tfidf_matrix_train = tfidf_vectorizer.fit_transform(text1)  #finds the tfidf score with normalization
ans1=cosine_similarity(tfidf_matrix_train[0:6], tfidf_matrix_train) #here the first element of tfidf_matrix_train is matched with other three elements



print(cosine_similarity(tfidf_matrix_train[0:2], tfidf_matrix_train))  #here the first element of tfidf_matrix_train is matched with other three elements
print(cosine_similarity(tfidf_matrix_train[0:3], tfidf_matrix_train))  #here the first element of tfidf_matrix_train is matched with other three elements
print(cosine_similarity(tfidf_matrix_train[0:4], tfidf_matrix_train))  #here the first element of tfidf_matrix_train is matched with other three elements
print(cosine_similarity(tfidf_matrix_train[0:5], tfidf_matrix_train))  #here the first element of tfidf_matrix_train is matched with other three elements
#print(cosine_similarity(tfidf_matrix_train[0:1], tfidf_matrix_train))  #here the first element of tfidf_matrix_train is matched with other three elements

ans2=pairwise_distances(tfidf_matrix_train[0:6],tfidf_matrix_train,metric='manhattan')