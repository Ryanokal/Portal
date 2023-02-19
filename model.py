import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.feature_selection import SelectKBest, f_classif

# Input data files are available in the read-only "../input/" directory

import warnings

# ignore warnings
warnings.filterwarnings("ignore")

# read csv (comma separated value) into data
data = pd.read_csv(r'E:\Python Projects\University Student Prediction/dataset.csv')
print(plt.style.available)  # look at available plot styles
plt.style.use('ggplot')

data.head()

data.info()

data.describe()

data.drop(data.index[(data["Target"] == "Other")], axis=0, inplace=True)

color_list = ["red" if i == "Dropout" else "green" for i in data.loc[:, "Target"]]
pd.plotting.scatter_matrix(data.loc[:, data.columns != "class"],
                           c=color_list,
                           figsize=[15, 15],
                           diagonal="hist",
                           alpha=0.5,
                           s=200,
                           marker="*",
                           edgecolor="black")
plt.show()

sns.countplot(x="Target", data=data)
data.loc[:, 'Target'].value_counts()

data.drop(data.index[(data["Target"] == "Enrolled")], axis=0, inplace=True)

sns.countplot(x="Target", data=data)
data.loc[:, 'Target'].value_counts()

# Feature Selection
X, y = data.loc[:, data.columns != 'Target'], data.loc[:, 'Target']
selector = SelectKBest(f_classif, k=10)
selector.fit(X, y)
X_new = selector.transform(X)
print(f"Feature Selection : {X.columns[selector.get_support()]}")

# KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
x, y = data.loc[:, data.columns != 'Target'], data.loc[:, 'Target']
knn.fit(x, y)
prediction = knn.predict(x)
print('Prediction: {}'.format(prediction))

# train test split
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
knn = KNeighborsClassifier(n_neighbors=3)
x, y = data.loc[:, data.columns != 'Target'], data.loc[:, 'Target']
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
# print('Prediction: {}'.format(prediction))
print("KNN\n")
print('With KNN (K=3) accuracy is: ', knn.score(x_test, y_test))  # ACCURACY = 0.83
print(f"F1 Score: {f1_score(y_test, knn.predict(x_test), average='macro'):.2f}")
print(f"Recall: {recall_score(y_test, knn.predict(x_test), average='macro'):.2f}\n")

# Model complexity
neig = np.arange(1, 25)
train_accuracy = []
test_accuracy = []
test_recall = []
train_recall = []
test_f1 = []
train_f1 = []

# Loop over different values of k
for i, k in enumerate(neig):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x_train, y_train)
    # train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))
    # train recall
    train_recall.append(knn.score(x_train, y_train))
    # test recall
    test_recall.append(knn.score(x_test, y_test))
    # train f1 score
    train_f1.append(knn.score(x_train, y_train))
    # test f1 score
    test_f1.append(knn.score(x_test, y_test))


# Plot
plt.figure(figsize=[13, 8])
plt.plot(neig, test_accuracy, label='Testing Accuracy')
plt.plot(neig, train_accuracy, label='Training Accuracy')
plt.plot(neig, test_f1, label='Testing F1 Score')
plt.plot(neig, train_f1, label='Training F1 Score')
plt.plot(neig, test_recall, label='Testing Recall')
plt.legend()
plt.title('-value VS Accuracy, Recall, and F1 Score')
# plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Metric Score')
# plt.ylabel('Accuracy')
plt.xticks(neig)
plt.savefig('graph.png')
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy), 1 + test_accuracy.index(np.max(test_accuracy))))
print("Best f1 score is {} with K = {}".format(np.max(test_f1), 1 + test_f1.index(np.max(test_f1))))
print("Best recall is {} with K = {}\n".format(np.max(test_recall), 1 + test_recall.index(np.max(test_recall))))

# Logistic regression - accuracy = 91.83%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(x_train, y_train)

pred = lr_clf.predict(x_test)
print("LOGISTIC REGRESSION")
print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
print(f"F1 Score: {f1_score(y_test, pred, average='macro'):.2f}")
print(f"Recall: {recall_score(y_test, pred, average='macro'):.2f}\n")

# Support Vector Machine (SVM)- accuracy  = 60.79%
from sklearn.svm import SVC

svm_clf = SVC(kernel='rbf', gamma=0.1, C=1.0)
svm_clf.fit(x_train, y_train)

pred = svm_clf.predict(x_test)
print("SUPPORT VECTOR MACHINE")
print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
print(f"F1 Score: {f1_score(y_test, pred, average='macro'):.2f}")
print(f"Recall: {recall_score(y_test, pred, average='macro'):.2f}\n")

# Decision Tree - accuracy  = 86.96%
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(random_state=40)
tree_clf.fit(x_train, y_train)

pred = tree_clf.predict(x_test)
print("DECISION TREE")
print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
print(f"F1 Score: {f1_score(y_test, pred, average='macro'):.2f}")
print(f"Recall: {recall_score(y_test, pred, average='macro'):.2f}")

# save the model to disk
joblib.dump(tree_clf, "student_model.sav")
