# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 22:46:56 2020

@author: kshit
"""

import nltk
from nltk.corpus import names
import random

def gender_features(word):
    return {'First_2letter': word[0:2]}



male_names = [(name, 'male') for name in names.words('male.txt')]
female_names = [(name, 'female') for name in names.words('female.txt')]
labeled_names = male_names + female_names
random.shuffle(labeled_names)
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
#entries are    ({'last_letter': 'g'}, 'male')
train_set, test_set = featuresets[500:], featuresets[:500]
print(gender_features('Shrek'))
classifier = nltk.NaiveBayesClassifier.train(train_set)

ans1 = classifier.classify(gender_features('Mark'))
ans2 = classifier.classify(gender_features('Precilla'))
classifier.show_most_informative_features(100)
print("Mark is:", ans1)
print("Precilla is:", ans2)

print(nltk.classify.accuracy(classifier, test_set))