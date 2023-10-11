import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep = ';')

# train/test split the features and response column
y = wine['quality'].values
X = wine.drop('quality', axis = 1).values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 1234)

print('dimension of the dataset: ', wine.shape)

np.random.seed(1)
nums = np.arange(1, 21)

class RandomForest:
    """
    Regression random forest using scikit learn's 
    decision tree as the base tree
    
    Parameters
    ----------
    n_estimators: int 
        the number of trees that you're going built
        on the bagged sample (you can even shutoff
        the bagging procedure for some packages)
    
    max_features: int 
        the number of features that you allow
        when deciding which feature to split on 
    
    all the other parameters for a decision tree like
    max_depth or min_sample_split also applies to Random Forest, 
    it is just not used here as that is more
    related to a single decision tree
    """

    def __init__(self, n_estimators, max_features):
        self.n_estimators = n_estimators
        self.max_features = max_features
        
    def fit(self, X, y):
        # for each base-tree models:
        # 1. draw bootstrap samples from the original data
        # 2. train the tree model on that bootstrap sample, and
        #    during training, randomly select a number of features to 
        #    split on each node
        self.estimators = []
        for i in range(self.n_estimators):
            boot = np.random.choice(y.shape[0], size = y.shape[0], replace = True)
            X_boot, y_boot = X[boot], y[boot]
            tree = DecisionTreeRegressor(max_features = self.max_features)
            tree.fit(X_boot, y_boot)
            self.estimators.append(tree)
            
        return self

    def predict(self, X):
        # for the prediction, we average the
        # predictions made by each of the bagged tree
        pred = np.empty((X.shape[0], self.n_estimators))
        for i, tree in enumerate(self.estimators):
            pred[:, i] = tree.predict(X)
            
        pred = np.mean(pred, axis = 1)
        return pred
    
# random forest using a random one third of the features at every split
rf = RandomForest(n_estimators = 50, max_features = 1 / 3)
rf.fit(X_train, y_train)
rf_y_pred = rf.predict(X_test)
print("MSE measures how far predictions deviate from target values. Smaller MSE values are better, and 0.0 means perfect prediction.")
print('Random Forest MSE',  mean_squared_error(y_test, rf_y_pred))