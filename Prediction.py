from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.svm import SVR


def prediction (dates,prices):

    X=dates.as_matrix().reshape(len(dates), 1)
    y = prices.as_matrix().reshape(len(prices), 1).ravel()
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    model= SVR(kernel='rbf', C=1e3, gamma=0.1)
    y_rbf = svr_rbf.fit(X, y).predict(X)
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3)





    ##print y_rbf
    lw = 2
    plt.scatter(X, y, color='darkorange', label='data')
    plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')

    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Regression')
    plt.legend()
    plt.show()
    return
