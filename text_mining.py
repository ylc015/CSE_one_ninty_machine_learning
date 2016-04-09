import numpy as np
import urllib
import scipy.optimize
import random
from collections import defaultdict
import nltk
import string
from nltk.stem.porter import *
from sklearn import linear_model
from nltk.corpus import stopwords
import unicodedata


def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("wine_review.json"))
print "done"


training_set = data[:50000]
testing_set = data[50000:100000]
print len(training_set)
print len(testing_set)


stopwords = stopwords.words('english')
wordCount = defaultdict(int)
punctuation = set(string.punctuation)
cat = defaultdict(int)
year = defaultdict(int)

#building bag of words
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation and not c.decode('utf-8','ignore') in stopwords])
  for w in r.split():
		wordCount[w] += 1

for d in data:
	cat[d['wine/variant']] += 1
	year[d['wine/year']] += 1

#100 most frequently appear words
counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

words = [x[1] for x in counts[:100]]
print words

#100 most frequently appear categories
counts = [(cat[w], w) for w in cat]
counts.sort()
counts.reverse()

cats = [x[1] for x in counts[:100]]

#100 most frequently appear make years
counts = [(year[w], w) for w in year]
counts.sort()
counts.reverse()

years = [x[1] for x in counts[:100]]




print "done find words"


### Sentiment analysis

wordId = dict(zip(words, range(len(words))))
wordSet = set(words)

catId = dict(zip(cats, range(len(cats))))
catSet = set(cats)

yearId = dict(zip(years, range(len(years))))
yearSet = set(years)

#building features
def feature(datum):
	feat = [0]*len(words)
	r = ''.join([c for c in datum['review/text'].lower() if not c in punctuation and not c.decode('utf-8', 'ignore') in stopwords])
	for w in r.split():
		if w in words:
			feat[wordId[w]] += 1

	#wine
	for i in range(len(cats)): feat.append(0)
	if d['wine/variant'] in cats:
		feat[catId[d['wine/variant']]] += 1

	#year
	for i in range(len(years)): feat.append(0)
	if d['wine/year'] in years:
		feat[yearId[d['wine/year']]] += 1

	feat.append(len(d['review/text']))

	feat.append(1) #offset
	return feat

X = [feature(d) for d in training_set]
y = [d['review/points'] for d in training_set]

print "done building feature" 

#No regularization
#theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

#With regularization
clf = linear_model.Ridge(1.0, fit_intercept=False)
clf.fit(X, y)
theta = clf.coef_
predictions = clf.predict(X)

def roundUp(p):
	if p > 100:
		p = 100
	elif p < 0:
		p = 0
	else:
		p = round(p)
	return p

print "linear regression"
predictions = [roundUp(p) for p in predictions]
y = [d['review/points'] for d in testing_set]
diff = np.array(predictions) - np.array(y)
mse = sum([ x * x for x in diff])/len(diff)
print mse


from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
"""
model = LogisticRegression()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print "logistic regression"
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
predictions = [roundUp(p) for p in predicted]
y = [d['review/points'] for d in testing_set]
diff = np.array(predictions) - np.array(y)
mse = sum([ x * x for x in diff])/len(diff)
print mse

print "navie bayes"
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
predictions = [roundUp(p) for p in predicted]
y = [d['review/points'] for d in testing_set]
diff = np.array(predictions) - np.array(y)
mse = sum([ x * x for x in diff])/len(diff)
print mse


print "KNN"
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
predictions = [roundUp(p) for p in predicted]
y = [d['review/points'] for d in testing_set]
diff = np.array(predictions) - np.array(y)
mse = sum([ x * x for x in diff])/len(diff)
print mse"""



print "classification and regression trees"
"""from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))# fit a k-nearest neighbor model to the data
predictions = [roundUp(p) for p in predicted]
y = [d['review/points'] for d in testing_set]
diff = np.array(predictions) - np.array(y)
mse = sum([ x * x for x in diff])/len(diff)
print mse


print "SVM"
from sklearn.svm import SVC
model = SVC()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
predictions = [roundUp(p) for p in predicted]
y = [d['review/points'] for d in testing_set]
diff = np.array(predictions) - np.array(y)
mse = sum([ x * x for x in diff])/len(diff)
print mse
"""




