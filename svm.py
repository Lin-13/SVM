from sklearn import svm
from sklearn import linear_model
import numpy as np
import pylab as pl
x=np.array([[2,0],[1,1],[2,3]])
y=np.array([0,0,1])
clf1=svm.SVC(kernel="linear")
clf1.fit(x,y)
print(clf1)
print(clf1.support_vectors_)
print(clf1.support_)
print(clf1.n_support_)
print(clf1.predict(np.array([[2,0],[10,10],[0,0]])))
