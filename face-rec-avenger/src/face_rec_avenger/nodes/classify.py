import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
    

def classify(X_train, y_train, X_test, y_test):
    
    # Split the data into training and testing sets.
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    # Convert the data to NumPy arrays.
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Train a SVM classifier on the training set.
    svm = SVC(kernel="linear")
    svm.fit(X_train, y_train)

    return svm