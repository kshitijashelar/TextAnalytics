# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 09:14:54 2020

@author: kshit
"""
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

doc=["Climate change is extreme crisis today with drastic change in climate","The extreme impact of climate change have been detected  on the globe the years","The artic is melting ice capades are melting at a  increase rate bad climate","drastic changes in extreme of climate such as decreases in Arctic sea ice","It is projected that drastic temeprature difference and climate change","The climate winter precipitation is projected to increase but decrease in summer precipitation are also projected","climate extremes are projected to become more intense as a consequence of climate projected that increase daily temperature","extreme precipitation  are projected to increase as a consequence of climate change","buildings across globe are expected to be exposed to drastic different climate conditions and extreme events","Failure for the movies of climate to the movies of movies"]


tknzr = TweetTokenizer()
tokenize_docs = []
for word in doc:
    tokenize_docs.append(tknzr.tokenize(str(doc)))
    
print(tokenize_docs)

filtered_doc_list=[]
stop_words = stopwords.words('english')
for current_list in tokenize_docs:
    temp_list=[]
    for word in current_list:
        if word not in stop_words:
            temp_list.append(word)
    filtered_doc_list.append(temp_list)
print(filtered_doc_list)

import re
for i in range (0, len(filtered_doc_list)):
    for j in range (0, len(filtered_doc_list[i])):
        filtered_doc_list[i][j] = re.sub(r"[^a-zA-Z0-9]+", '', filtered_doc_list[i][j])
        filtered_doc_list[i][j] = filtered_doc_list[i][j].lower()
        

print(filtered_doc_list)


final_doc=[]
for cur_list in filtered_doc_list:
    temp_list=[]
    temp_list = [w for w in cur_list if w]
    final_doc.append(temp_list)
print(final_doc)

from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

tagged_docs=[]
for cur_lst in final_doc:
    tagged_docs.append(pos_tag(cur_lst))
print(tagged_docs)

wnlm = WordNetLemmatizer()
def convert_big_tags(t):
    if t=='vbd' or t=='vbg' or t=='vbz' or t=='vbp' or t=='vbn':
        return 'v'
    elif t == 'jj' or t == 'jjr' or t=='jjs':
        return 'a'
    elif t == 'rb' or t=='rbr' or t=='rbs':
        return 'r'
    else:
        return 'n'
lemmatized_docs=[]
for cur_doc in tagged_docs:
    temp_docs=[]
    for word in cur_doc:
        new_tag = convert_big_tags(word[1].lower())
        output = wnlm.lemmatize(word[0], new_tag)
        temp_docs.append(output)
    lemmatized_docs.append(temp_docs)
print(lemmatized_docs)

final_output = []
for ele in lemmatized_docs:
    string_join = ' '.join(ele)
    final_output.append(string_join)
print(final_output)

joined_lst=""
for ele in final_output:
    new_line = ''.join(ele)
    joined_lst= joined_lst+new_line
print(joined_lst)

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
cnt_vec =  CountVectorizer()
tf_ob = cnt_vec.fit_transform(final_output)
tf_df = pd.DataFrame(tf_ob.todense(), columns = cnt_vec.get_feature_names())
tf_df


from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords = stop_words, 
                min_font_size = 8).generate(joined_lst)
plt.imshow(wordcloud,cmap=plt.cm.gray, interpolation='bilinear')
plt.axis('off')
plt.show()