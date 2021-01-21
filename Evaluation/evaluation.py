# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:21:11 2020

@author: kshit
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

path = r"C:\Users\kshit\OneDrive\Documents\Text Analytics\Book2.xlsx"
df = pd.read_excel(path)
print(df)


precision=[]
recall = []
fpr = []
fnr = []
f1 = []
for i in range(len(df)): 
    precision.append(df.iloc[i]['TP']/(df.iloc[i]['TP'] + df.iloc[i]['FP']))
    recall.append(df.iloc[i]['TP']/(df.iloc[i]['TP'] + df.iloc[i]['FN']))
    fpr.append(df.iloc[i]['FP']/(df.iloc[i]['FP'] + df.iloc[i]['TN']))
    fnr.append(df.iloc[i]['FN']/(df.iloc[i]['FN'] + df.iloc[i]['TP'])) #or 1-TPR
    f1.append((2*precision[i]*recall[i])/(precision[i]+recall[i]))
    
    
    

df['Precision'] = precision
df['Recall/TPR'] = recall
df['FPR'] = fpr
df['FNR/Miss-Detection'] = fnr
df['F1-Measure'] = f1

print(df)




plt.figure(figsize=(10,9),dpi=150)
plt.plot(fpr, recall, color='red',
         lw=2, label='ROC curve', marker="X")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
for x,y in zip(fpr,recall):
    label = '('+str("{:.2f}".format(x))+','+str("{:.2f}".format(y))+')'
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()


fig,ax = plt.subplots(figsize=(10,9),dpi=150)
plt.plot(fpr, fnr, color='blue',
         lw=2, label='DET curve', marker="X")

ticks_to_use = [0.001,0.01,0.1,0.2,1,5,10]

plt.yscale('log')
plt.xscale('log')

ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax.set_xticks(ticks_to_use)
ax.set_yticks(ticks_to_use)



for x,y in zip(fpr,fnr):
    label = '('+str("{:.2f}".format(x))+','+str("{:.2f}".format(y))+')'
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

    
plt.xlabel('False Positive Rate')
plt.ylabel('False Negative Rate')
plt.title('DET Curve')
plt.legend(loc="upper right")
plt.show()