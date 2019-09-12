import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('winequality-red.csv')

df.info()

cor = df.corr()
cor['quality'].sort_values(ascending = False)

df = df.drop(['residual sugar','fixed acidity', 'free sulfur dioxide','pH','chlorides', 'density'], axis = 1)



col = ['alcohol', 'sulphates', 'citric acid', 'total sulfur dioxide', 'volatile acidity']
for char in col:
    sns.barplot(df['quality'], df[char], palette = 'Reds')
    plt.show()
    
df.describe()


df[df['quality'] == 3].count()
df[df['quality'] == 5].count()
df[df['quality'] == 6].count()
df[df['quality'] == 4].count()

df1 = df[df['quality'] == 3]
df2 = df[df['quality'] == 5]
df3 = df[df['quality'] == 6]
df4 = df[df['quality'] == 7]
df5 = df[df['quality'] == 8]
df6 = df[df['quality'] == 4]


plt.scatter(x = df1['volatile acidity'], y = df1['alcohol'], marker = 'o', color = 'red')

plt.scatter(x = df2['volatile acidity'], y = df2['alcohol'], marker = '^', color =  'green')

plt.scatter(x = df3['volatile acidity'], y = df3['alcohol'], marker = '*')

plt.scatter(x = df4['quality'], y = df4['alcohol'], marker = 'o')

plt.scatter(x = df5['quality'], y = df5['alcohol'], marker = 'o')

plt.scatter(x = df6['quality'], y = df6['alcohol'], marker = 'o')

plt.scatter(x = df1['quality'], y = df1['alcohol'], marker = 'o')




x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.cross_validation import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf = 3,max_features = 'auto' )
model.fit(x_train, y_train)


y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)



from sklearn.svm import SVC
model1 = SVC(kernel = 'sigmoid')
model1.fit(x_train, y_train)


y_pred1 = model1.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred1)


df10 = df

category = []
for i in df['quality']:
    if i<5:
        category.append('Bad')
    elif i > 6:
        category.append('Good')
    else:
        category.append('Medium')
        
category = pd.DataFrame(data = category, columns = ["category"])
df10 = pd.concat([df10,category], axis = 1)

df10 = df10.drop(['quality'], axis=  1)

x1 = df10.iloc[:, :-1].values
y1 = df10.iloc[:, -1].values

from sklearn.cross_validation import train_test_split
x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size = 0.3)

dfa = df[df['category'] == 'Good']
dfb = df[df['category'] == 'Bad']
dfc = df[df['category'] == 'Medium']



from sklearn.tree import DecisionTreeClassifier
model10 = DecisionTreeClassifier(criterion='entropy')
model10.fit(x_train1, y_train1)


plt.figure(figsize = (20, 12))
plt.scatter(x = dfa['volatile acidity'], y = dfa['alcohol'], marker = 'o', color = 'red')

plt.scatter(x = dfb['volatile acidity'], y = dfb['alcohol'], marker = '^', color =  'green')

plt.scatter(x = dfc['volatile acidity'], y = dfc['alcohol'], marker = '*', color = 'black')
plt.show()

y_pred10 = model10.predict(x_test1)


from sklearn.metrics import accuracy_score
accuracy_score(y_test1, y_pred10)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm = 'auto', metric = 'minkowski')
knn.fit(x_train1, y_train1)

y_predknn = knn.predict(x_test1)


from sklearn.metrics import accuracy_score
accuracy_score(y_predknn, y_test1)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_predknn, y_test1)