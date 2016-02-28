import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

weightdata = pd.read_csv('data.csv')

#weightdata.to_csv('weightData_clean.csv', header=True, index=False)

#rename columns to remove single quotes that were originally in them
weightdata.columns = ['id', 'sex', 'actual', 'ideal', 'diff']

#remove single quotes surrounding sex
#weightdata['sex'] = weightdata['sex'].map(lambda x: str(x.rstrip("'")))
#weightdata['sex'] = weightdata['sex'].map(lambda x: str(x.lstrip("'")))

#plot actual and ideal weight on a chart
plt.hist(weightdata['actual'],histtype='bar')
plt.hist(weightdata['ideal'],histtype='bar')
#plt.show()

#plot difference in weight on a chart
plt.hist(weightdata['diff'],histtype='bar')
#plt.show()

#map sex to a categorical variable
weightdata['sexcat'] = weightdata['sex'].astype('category')


#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
#This is definitely wrong but I'm just trying to get the model to run at all, before trying to get everything else in there.
y = weightdata['sexcat']
X = weightdata[['actual', 'ideal', 'diff']]
clf = GaussianNB()
clf.fit(X, y) #need a predict method (using same X) and see from this prediction how many Y's are diff than Y actual
print(clf.predict([[145, 160, -15]]))
print(clf.predict([[160, 145, 15]]))

y_pred = clf.fit(X, y).predict(X)
print("Number of mislabeled points out of a total %d points : %d" % (X.shape[0],(y != y_pred).sum()))

#y_pred lines come from http://scikit-learn.org/stable/modules/naive_bayes.html
