import numpy as np

# 1D linear fit
def ls_fit_1d(x, y, max_deg):
    # X matrix
    X = np.ones((len(x), 1))
    for i in range (1, max_deg+1):
        X = np.hstack((X, x ** i))
    # Calculating beta_hat using the normal equations
    beta_hat = np.matmul(np.transpose(X), X) 
    beta_hat = np.linalg.inv(beta_hat)
    beta_hat = np.matmul(beta_hat, np.transpose(X))
    beta_hat = np.matmul(beta_hat, y)
    return beta_hat, X

