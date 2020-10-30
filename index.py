import pandas as pd
import sklearn as sk
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

### Prepare the Data ###

headings = ["calorific_value", "nitrogen", "turbidity", "style", "alcohol",
            "sugars", "bitterness", "beer_id", "colour", "degree_of_fermentation"]

# Read the data
beer_training = pd.read_csv('beer_training.txt', sep="\t", header=None)
beer_test = pd.read_csv('beer_test.txt', sep="\t", header=None)

# Apply headings to data
beer_training.columns = headings
beer_test.columns = headings

# Remove beer_id as is not needed
beer_training.drop('beer_id', axis=1, inplace=True)
beer_test.drop('beer_id', axis=1, inplace=True)

# set training and test x and y (features and label)
x_train = beer_training.loc[:, beer_training.columns != 'style'].values
y_train = beer_training['style']

x_test = beer_test.loc[:, beer_training.columns != 'style'].values
y_test = beer_test['style']

### Implement Algorithms - KNN and SVC ###

# init and fit the classifiers - trains the model
knn = KNeighborsClassifier(n_neighbors=2).fit(x_train, y_train)
svm_model_linear = SVC(kernel='linear', C=1).fit(x_train, y_train)

# get accuracy using test and training set

#knn - test
knn_pred = knn.predict(x_test)
knn_accuracy = knn.score(x_test, y_test)
print("KNN Test Accuracy: ", knn_accuracy)

#knn - training
knn_accuracy = knn.score(x_train, y_train)
print("KNN Train Accuracy", knn_accuracy)

#svm - test
svm_pred = svm_model_linear.predict(x_test)
svm_accuracy = svm_model_linear.score(x_test, y_test)
print("SVM Test Accuracy: ", svm_accuracy)

#svm - training
svm_accuracy = svm_model_linear.score(x_train, y_train)
print("SVM Train accuracy: ", svm_accuracy)

### Print out Classifier Metrics ###

# confusion matrix (for test)

print("\n *** The SVM Confusion Matrix *** \n")
print(confusion_matrix(svm_pred, y_test))

print("\n *** The KNN Confusion Matrix *** \n")
print(confusion_matrix(knn_pred, y_test))

# classification report (for test)

print("\n*** SVM Classification Report ***\n")
print(classification_report(svm_pred, y_test))

print("\n*** KNN Classification Report ***\n")
print(classification_report(knn_pred, y_test))
