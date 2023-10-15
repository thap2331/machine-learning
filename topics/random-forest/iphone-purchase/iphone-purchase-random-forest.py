import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv('https://raw.githubusercontent.com/omairaasim/machine_learning/master/project_16_random_forest_classifier/iphone_purchase_records.csv', sep = ',')
y = data['Purchase Iphone'].values
X = data.drop('Purchase Iphone', axis = 1).values

# Convert Gender to number
labelEncoder_gender =  LabelEncoder()
X[:,0] = labelEncoder_gender.fit_transform(X[:,0])

# train/test split the features and response column
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.25, random_state = 1234)

np.random.seed(1)

class RandomForest:
    """
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

    def __init__(self, n_estimators, max_features=1/3):
        self.n_estimators = n_estimators
        self.max_features = max_features
        
    def fit(self, X_train, y_train):
        classifier = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0, 
                                            max_features=self.max_features)
        classifier.fit(X_train, y_train)

        return classifier        
    
# random forest using a random one third of the features at every split
rf = RandomForest(n_estimators = 50, max_features = 1 / 3)
classifier = rf.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

confusion_matrix_table = metrics.confusion_matrix(y_test, y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred) 
precision = metrics.precision_score(y_test, y_pred) 
recall = metrics.recall_score(y_test, y_pred) 
print('accuracy:', accuracy, '\nprecision:', precision, '\nrecall:', recall)
print(confusion_matrix_table)