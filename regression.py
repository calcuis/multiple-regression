import numpy as np

def multiple_regression(X, y):
    # Add a column of ones to X to account for the bias term
    X = np.column_stack((np.ones(X.shape[0]), X))

    # Compute the Moore-Penrose pseudoinverse of X
    X_pinv = np.linalg.pinv(X)

    # Compute the coefficients of the regression model
    coef = X_pinv.dot(y)

    return coef
