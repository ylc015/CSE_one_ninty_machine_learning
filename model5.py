import numpy
import urllib
import scipy.optimize
import random
from sklearn.decomposition import PCA
from collections import defaultdict
import gzip
import math


#question 2
def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

count = 0
totalPurchases = 0 
itemCount = defaultdict(int)
perferences = defaultdict(list)
categories = defaultdict(list)
user_pur = defaultdict(list)
item_user = defaultdict(list)
user_av_rating = defaultdict(list)
all_user_item = []
for l in readGz("train.json.gz"):
    user,item, rating = l['reviewerID'],l['itemID'], l['rating']
    for c in l['category']:
    	perferences[user].append(c)
        categories[item].append(c)
    user_pur[user].append(item)
    item_user[item].append(user)
    itemCount[item] += 1
    totalPurchases += 1


mostPopular = [(itemCount[x], x) for x in itemCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
  count += ic
  return1.add(i)
  if count > totalPurchases/4: break


predicted_true = 0
predicted_false = 0
    
count = 0
predictions = open("predictions_Purchase.txt", 'w')
for l in open("pairs_Purchase.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue

  u,i = l.strip().split('-')
  
  list_item = []
  for j in user_pur[u]:
    list_item.append(j)

  

  item = []
  for k in categories[i]:
    for k1 in k:
      item.append(k1)

  item_before = []
  for l in list_item:
    for n in categories[l]:
      for n1 in n:
        item_before.append(n1)


  intersect = list(set(item_before) & set(item))
  union = list(set(item_before) | set(item))

  if (len(intersect) == 0) or (len(union) == 0):
    ratio = 0.0
  else:
    ratio = (len(intersect) * 1.0) / len(union)  

  if i in return1:
    predictions.write(u + '-' + i + ",1\n")
    predicted_true += 1

  elif u in user_pur :
      predictions.write(u + '-' + i + ",1\n")
      predicted_true += 1
      print "new user"
  elif i in item_user:
      predictions.write(u + '-' + i + ",1\n")
      predicted_true += 1
      print "new item"

  elif i not in return1:
    if ratio > 0.5 :
      predictions.write(u + '-' + i + ",1\n")
      predicted_true += 1
      print "not popular but similar"

    else:
      predictions.write(u + '-' + i + ",0\n")
      predicted_false += 1
      print "not popular not similar"


 


  count += 1


print "predict T: %d F: %d " % (predicted_true, predicted_false)
predictions.close()    
