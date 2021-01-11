# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 10:36:09 2020

@author: kshit
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 09:21:18 2020

@author: kshit
"""
spam_tweets=[
    "#BurgerKing Super Spicy Chicken Whopper : Enjoy the taste of exquisite Fresh Chicken with lettuce mayonise onions tomatoes and cheese!! Just for 3 Euros! Burgers never tasted so good!",
    "#BurgerKing Triple Beacon Cheeseburger Whopper: Who says you can not enjoy a triple beacon cheeseburger in 2.99? Enjoy the taste of burger overloaded with fresh meat and veggies with taste of delicious beverages",
    "With #BurgerKing hot coffee, get ready to face the day with super energy! #HappyHalloween",
    "#BurgerKing Hot and crispy Fries : Ready for a long weekend shopping with friends? Enjoy the hot and crispy fries with your burger and a beverage! Special weekend offer on meals!",
    "#BurgerKing spicy hot chicken wings : For those who prefer the crunchy texture of fried chicken, these wings delivers a true punch of flavours with fresh irish chicken and total bliss",
    "#BurgerKing Hot Chocolate Fudge: Craving a nice warm bowl of hot chocolate fudge? head over to your nearest BurgerKing outlet and try put new addition in the menu with chocolate fudge",
    "With #BurgerKing hot coffee, get ready to face the day with super energy! #HappyHalloween",
    "#BurgerKing Fish n Chips: Enjoy the perfect blend of yummy salmon batter fried with BurgerKing's hot beverages",
    "Treat the desi in you with #BurgerKing Happy Treats chicken tikka masala burger. Keep all eyes on the food, just on the food.",
    "With #BurgerKing hot coffee, get ready to face the day with super energy! #HappyHalloween",
    "#BurgerKing special hamburgers : Enjoy the authentic taste of burger with lettuce mayonise onions tomatoes and cheese!! Just for 3 Euros! Burgers never tasted so good!"
]

print(spam_tweets)
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import nltk

tknzr = TweetTokenizer()
tokenizeSpamTweets = []
for tweet in spam_tweets:
    tokenizeSpamTweets.append(tknzr.tokenize(tweet))

#Removing stop words
filteredSpamTweet=[]
stop_words = stopwords.words('english')
for current_list in tokenizeSpamTweets:
    temp_list=[]
    for word in current_list:
        if word not in stop_words:
            temp_list.append(word)
    filteredSpamTweet.append(temp_list)

#Removing the special characters and changing to lower case
import re
for i in range (0, len(filteredSpamTweet)):
    for j in range (0, len(filteredSpamTweet[i])):
        filteredSpamTweet[i][j] = re.sub(r"[^a-zA-Z0-9]+", '', filteredSpamTweet[i][j])
        filteredSpamTweet[i][j] = filteredSpamTweet[i][j].lower()
        
finalSpamTweet=[]
for cur_list in filteredSpamTweet:
    temp_list=[]
    temp_list = [w for w in cur_list if w]
    finalSpamTweet.append(temp_list)

finalOutputSpam=[]
for lst in finalSpamTweet:
    for ele in lst:
        finalOutputSpam.append(ele)
        
print(finalOutputSpam)

import math
def entropy(labels):
    freqdist = nltk.FreqDist(labels)
    probs = [freqdist.freq(l) for l in freqdist]
    return -sum(p * math.log(p,2) for p in probs)


entropy_spam = entropy(finalOutputSpam)
print(entropy_spam)

random =["Arnab Goswami reaches court and pleads not guilty for the TRP charge #arnab#newstrp#newschannel#justice#mumbai",
         "WHO chairman says that we have no idea how COVID-19 spreads. Cites new research.#who#covid#healthcare#eachforthemselves",
         "#proposalgoeswrong#weddingproposal#ring#care",
        "Dublin put under another lockdown.Indian students repent the decision to go there as they are forced to attend classes from hostel rooms.#dublin#covid#lockdown#studentsproblems",
        "Indian student gets lost in university campus inspite of having google maps. University decides to put up signboards#university#signboards#dublin#rightdurections",
        "Recently released movie ‘Trial of Chicago 7’ breaks all streaming records. Aaron Sorkin wins big #netflix#movie#sorkin#newrelease",
        "Police in Ghana make a thief eat 2 dozen bananas after swallowing a stolen gold chain. #policetechniques#newways#modernproblems#modernsolutions",
        "Scientists construct a new all inclusive protective shield to counter polution. Public says it makes them feel like astronauts on earth. #toomuchscience#advancedtech#publicrhetoric",
        "IPL commences in Dubai amid empty stadiums scares. Most of the time is lost in searching the ball when hit for a six and it falls into the stands. #ipl#cricket#emptystands",
        "Man tweets against a popular eatery for not serving the porridge hot A fake account handler replied asking him to get his ‘Goldilocks Ass’ in early to get the porridge hot. #trollrers#porridge#hot#consumerworries"
    ]

print(random)

tknzr = TweetTokenizer()
tokenizeRandomTweets = []
for tweet in random:
    tokenizeRandomTweets.append(tknzr.tokenize(tweet))
    
#Removing stop words
filteredRandomTweet=[]
stop_words = stopwords.words('english')
for current_list in tokenizeRandomTweets:
    temp_list=[]
    for word in current_list:
        if word not in stop_words:
            temp_list.append(word)
    filteredRandomTweet.append(temp_list)

#Removing the special characters and changing to lower case
import re
for i in range (0, len(filteredRandomTweet)):
    for j in range (0, len(filteredRandomTweet[i])):
        filteredRandomTweet[i][j] = re.sub(r"[^a-zA-Z0-9]+", '', filteredRandomTweet[i][j])
        filteredRandomTweet[i][j] = filteredRandomTweet[i][j].lower()

finalRandomTweet=[]
for cur_list in filteredRandomTweet:
    temp_list=[]
    temp_list = [w for w in cur_list if w]
    finalRandomTweet.append(temp_list)
    
finalOutputRandomTweet=[]
for lst in finalRandomTweet:
    for ele in lst:
        finalOutputRandomTweet.append(ele)
        
print(finalOutputRandomTweet)

ent_random = entropy(finalOutputRandomTweet)
print(ent_random)


combinedTweets = finalOutputRandomTweet +finalOutputSpam


ent_cobined = entropy(finalOutputRandomTweet +finalOutputSpam)
print(ent_cobined)