
from sklearn.model_selection import KFold


def train_eval(model, X_train, X_test, y_train, y_test):
    
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    kf = KFold(n_splits=5, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Evaluate the accuracy of 
        # the classifier on the validation set
        acc = model.score(X_test, y_test)
        print(f"Train accuracy: {acc}")