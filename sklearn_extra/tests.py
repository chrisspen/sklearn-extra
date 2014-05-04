import unittest
import random

from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from ensemble import StreamingRandomForestClassifier, DecisionTreeClassifier

class Tests(unittest.TestCase):
    
    def test_forest(self):
        
        n_estimators=100
        
        print('DecisionTreeClassifier')
        random.seed(0)
        iris = load_iris()
        clf = DecisionTreeClassifier()
        scores = cross_val_score(clf, iris.data, iris.target)
        s1 = scores.mean()
        print('mean:',s1)
        self.assertTrue(s1 > 0.93)
        
        print('RandomForestClassifier')
        random.seed(0)
        iris = load_iris()
        clf = RandomForestClassifier(n_estimators=n_estimators)
        scores = cross_val_score(clf, iris.data, iris.target)
        s1 = scores.mean()
        print('mean:',s1)
        self.assertTrue(s1 > 0.93)
        
        print('AdaBoostClassifier')
        random.seed(0)
        iris = load_iris()
        clf = AdaBoostClassifier(n_estimators=n_estimators)
        scores = cross_val_score(clf, iris.data, iris.target)
        s2 = scores.mean()
        print('mean:',s2)
        self.assertTrue(s2 > 0.93)
        
        print('StreamingRandomForestClassifier')
        random.seed(0)
        iris = load_iris()
        clf = StreamingRandomForestClassifier(n_estimators=n_estimators)
        scores = cross_val_score(clf, iris.data, iris.target)
        s3 = scores.mean()
        print('mean:',s3)
        self.assertTrue(s3 > 0.93)

if __name__ == '__main__':
    unittest.main()
    