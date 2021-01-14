# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 15:56:22 2020

@author: kshit
"""

def JaccardDist(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    ans = 1 - float(len(set1 & set2))/len(set1|set2)
    return round(ans,2)


def DiceCoef(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    coef = float(2*len(set1 & set2))/(len(set1) + len(set2))
    return round(1-coef,2)
    
Chicken_nuggets = "spicy fried cheesy crispy meaty"
Cheesecake= "sweet baked cheesy creamy non-meaty"
Pie = "sour fried non-cheesy crumbly meaty"
Soup = "salty grilled cheesy creamy meaty"
Burgers = "spicy fried cheesy crispy meaty"

arr = [Chicken_nuggets,Cheesecake, Pie, Soup, Burgers]
food = ["Chicken_nuggets", "Cheesecake", "Pie", "Soup", "Burgers"]
ans = [[999,999,999,999,999],[999,999,999,999,999],[999,999,999,999,999],[999,999,999,999,999],[999,999,999,999,999]]
dice = [[999,999,999,999,999],[999,999,999,999,999],[999,999,999,999,999],[999,999,999,999,999],[999,999,999,999,999]]

ans1 = JaccardDist(Chicken_nuggets, Chicken_nuggets)
ans2 = JaccardDist(Chicken_nuggets, Cheesecake)
ans3 = JaccardDist(Chicken_nuggets, Pie)
ans4 = JaccardDist(Chicken_nuggets, Soup)
ans5 = JaccardDist(Chicken_nuggets, Burgers)



for i in range(0,5):
    for j in range(0,5):
        
        ans[i][j] = JaccardDist(arr[i],arr[j])
        dice[i][j] = DiceCoef(arr[i], arr[j])

#print("Chicken_nuggets,Cheesecake, Pie, Soup, Burgers")
print(food)
for i in range(0,4):
    for j in range(0,4):
        print(str(ans[i][j]) + str('\t'))
    print('\n')
        

        

