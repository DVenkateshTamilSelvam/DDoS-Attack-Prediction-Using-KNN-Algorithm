import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    cohen_kappa_score, accuracy_score, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay
)
import pickle
import warnings

# Ignore future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load and preprocess the dataset
dataset = pd.read_csv('ddos_dataset.csv')
dataset = dataset.dropna(how="any")

# Display the dataset and its info
print(dataset)
print(dataset.info())

# Plot a histogram of the 'attack' feature
plt.figure(figsize=(10, 8))
plt.title("Histogram of attack")
plt.hist(dataset['attack'], rwidth=0.9)
plt.show()

# Plot a bar graph of 'N_IN_Conn_P_SrcIP' against 'attack'
m = dataset['attack']
n = dataset['N_IN_Conn_P_SrcIP']
plt.figure(figsize=(4, 4))
plt.title("Bar plot graph", fontsize=20)
plt.xlabel("attack", fontsize=15)
plt.ylabel("N_IN_Conn_P_SrcIP", fontsize=15)
plt.bar(m, n, label="bar plot", color="orange", width=0.1)
plt.legend(loc='best')
plt.show()

# Select features and target
X = dataset.iloc[:, 6:16].values
y = dataset.iloc[:, 16].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=121)

# Display the training data
print(X_train)
print(" ")
print(y_train)

# Train the KNN model
print("Training Started")
print("Processing")
classifier = KNeighborsClassifier(n_neighbors=2, weights='distance', algorithm='brute', p=1, leaf_size=1)
classifier.fit(X_train, y_train)

# Save the trained model to disk
with open('knnpickle_file', 'wb') as knnPickle:
    pickle.dump(classifier, knnPickle)

print("Training Completed")

# Predict the test set results
y_pred = classifier.predict(X_test)

# Round the predictions
y_pred = y_pred.round()
print(y_pred)

# Calculate ROC curve and AUC score
fpr1, tpr1, _ = roc_curve(y_test, y_pred)
auc_score1 = auc(fpr1, tpr1)

# Compute and display the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Print evaluation metrics
print("KNN ALGORITHM")
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy Score: ', accuracy)

precision = precision_score(y_test, y_pred)
print('Precision: %f' % precision)

recall = recall_score(y_test, y_pred)
print('Recall: %f' % recall)

f1 = f1_score(y_test, y_pred)
print('F1 score: %f' % f1)

kappa = cohen_kappa_score(y_test, y_pred)
print('Cohen\'s kappa: %f' % kappa)

# Plot the ROC curve
plt.figure(figsize=(7, 6))
plt.plot(fpr1, tpr1, color='blue', label='ROC (KNN AUC = %0.4f)' % auc_score1)
plt.legend(loc='lower right')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()
