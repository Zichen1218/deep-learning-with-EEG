from mne.decoding import CSP

def fit_csp(X_train, y_train, X_test, n_components=4):
    """
    Fit CSP on training data and transform both train and test.
    
    Args:
        X_train:      np.ndarray, shape (n_train, n_channels, n_times)
        y_train:      np.ndarray, shape (n_train,)
        X_test:       np.ndarray, shape (n_test, n_channels, n_times)
        n_components: int, CSP components per binary subproblem
    
    Returns:
        X_train_csp: np.ndarray, shape (n_train, n_features)
        X_test_csp:  np.ndarray, shape (n_test, n_features)
        csp:         fitted CSP object
    """
    # YOUR CODE HERE
    csp = CSP(n_components=n_components,reg=None,log = True,transform_into='average_power')
    csp.fit(X_train,y_train)
    X_train_csp = csp.transform(X_train)
    X_test_csp = csp.transform(X_test)
    return X_train_csp,X_test_csp,csp