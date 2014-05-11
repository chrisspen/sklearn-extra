from __future__ import print_function
import unittest
import random
from functools import partial

from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris, load_digits
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from ensemble import (
    StreamingDecisionTreeClassifier,
    StreamingRandomForestClassifier,
    StreamingExtraTreesClassifier,
)

# Note, this must be globally accessible so it can be used with
# multiprocessing.
def test_map_train(cls, args):
    a,b = args
    print('training:',a,b)
    dataset = load_digits()
    data = [_ for i,_ in enumerate(dataset.data) if (i % b) == a]
    target = [_ for i,_ in enumerate(dataset.target) if (i % b) == a]
    clf = cls()
    clf.fit(data, target)
    score = clf.score(data, target)
    print('score:',score)
    return clf

class TestMapTrain(object):
    
    def train(self, cls, args):
        a,b = args
        print('training:',a,b)
        dataset = load_digits()
        data = [_ for i,_ in enumerate(dataset.data) if (i % b) == a]
        target = [_ for i,_ in enumerate(dataset.target) if (i % b) == a]
        clf = cls()
        clf.fit(data, target)
        score = clf.score(data, target)
        print('score:',score)
        return clf

class Tests(unittest.TestCase):
    
    def test_forest_classifiers(self):
        """
        Confirm the basic accuracy of our classifiers.
        """
        
        #http://scikit-learn.org/stable/datasets/
        n_estimators=100
        
        datasets = [
            ('Iris', load_iris()),
            ('Digits', load_digits()),
        ]
        
        classifiers = [
            (AdaBoostClassifier, partial(AdaBoostClassifier, n_estimators=n_estimators)),
            
            (ExtraTreesClassifier, partial(ExtraTreesClassifier, n_estimators=n_estimators)),
#            
            (DecisionTreeClassifier, DecisionTreeClassifier),
            (StreamingDecisionTreeClassifier, partial(StreamingDecisionTreeClassifier, n_estimators=n_estimators)),
#
            (RandomForestClassifier, partial(RandomForestClassifier, n_estimators=n_estimators)),
            (StreamingRandomForestClassifier, partial(StreamingRandomForestClassifier, n_estimators=n_estimators)),
#            
            (ExtraTreesClassifier, partial(ExtraTreesClassifier, n_estimators=n_estimators)),
            (StreamingExtraTreesClassifier, partial(StreamingExtraTreesClassifier, n_estimators=n_estimators)),
        ]
        
        for name, dataset in datasets:
            print('\nDataset\t%s' % name)
            print('Score\tClassifier')
            for cls, cls_callable in classifiers:
                random.seed(0)
                clf = cls_callable()
                scores = cross_val_score(clf, dataset.data, dataset.target)
                score = scores.mean()
                print('%.04f\t%s' % (score, cls.__name__))
    
    def test_reduce(self):
        """
        Confirm we can merge our estimators together.
        """
        
        scores = {}
        
        #http://scikit-learn.org/stable/datasets/
        n_estimators=100
        
        datasets = [
            ('Iris', load_iris()),
            ('Digits', load_digits()),
        ]
        
        classifiers = [
#            (AdaBoostClassifier, partial(AdaBoostClassifier, n_estimators=n_estimators)),
#            
#            (ExtraTreesClassifier, partial(ExtraTreesClassifier, n_estimators=n_estimators)),
##            
            (DecisionTreeClassifier, DecisionTreeClassifier),
            (StreamingDecisionTreeClassifier, partial(StreamingDecisionTreeClassifier, n_estimators=n_estimators)),

            (RandomForestClassifier, partial(RandomForestClassifier, n_estimators=n_estimators)),
            (StreamingRandomForestClassifier, partial(StreamingRandomForestClassifier, n_estimators=n_estimators)),
            
            (ExtraTreesClassifier, partial(ExtraTreesClassifier, n_estimators=n_estimators)),
            (StreamingExtraTreesClassifier, partial(StreamingExtraTreesClassifier, n_estimators=n_estimators)),
        ]
        
        for name, dataset in datasets:
            dlen = len(dataset.data)
            print('\nDataset\t%s' % name)
            print('Score\tClassifier')
            for cls, cls_callable in classifiers:
                random.seed(0)
                clf0 = cls_callable()
                
                # Train a reference classifier on the full dataset.
                clf0.fit(dataset.data, dataset.target)
                score0 = clf0.score(dataset.data, dataset.target)
                
                # Train a classifier on the first half of the dataset.
                clf_a = cls_callable()
                clf_a.fit(dataset.data[:dlen/2], dataset.target[:dlen/2])
                score_a = clf_a.score(dataset.data, dataset.target)
                
                # Train a classifier on the second half of the dataset.
                clf_b = cls_callable()
                clf_b.fit(dataset.data[dlen/2:], dataset.target[dlen/2:])
                score_b = clf_b.score(dataset.data, dataset.target)
                
                # Merge A+B and test on the full dataset to measure
                # the usefulness of the merge.
                score1 = 0.0
                if hasattr(cls, 'reduce'):
                    clf1 = cls.reduce(clf_a, clf_b)
                    score1 = clf1.score(dataset.data, dataset.target)
                
                scores['%s %s 0' % (name, cls.__name__)] = score0
                scores['%s %s A' % (name, cls.__name__)] = score_a
                scores['%s %s B' % (name, cls.__name__)] = score_b
                scores['%s %s 1' % (name, cls.__name__)] = score1
                
                print('%.04f\t%s 0' % (score0, cls.__name__))
                print('%.04f\t%s A' % (score_a, cls.__name__))
                print('%.04f\t%s B' % (score_b, cls.__name__))
                print('%.04f\t%s 1' % (score1, cls.__name__))
        
        self.assertEqual(scores['Digits StreamingExtraTreesClassifier 1'], 1.0)
        self.assertTrue(scores['Digits StreamingRandomForestClassifier 1'] > scores['Digits StreamingRandomForestClassifier A'])
        self.assertTrue(scores['Digits StreamingRandomForestClassifier 1'] > scores['Digits StreamingRandomForestClassifier B'])
    
    def test_pickle(self):
        """
        Confirm we can reliably pickle our estimators.
        """
        import pickle
        clf = StreamingExtraTreesClassifier()
        dataset = load_digits()
        clf.fit(dataset.data, dataset.target)
        score0 = clf.score(dataset.data, dataset.target)
        s = pickle.dumps(clf)
        clf = pickle.loads(s)
        score1 = clf.score(dataset.data, dataset.target)
        self.assertEqual(score0, score1)
    
    def test_map(self):
        """
        Confirm we can train forests in parallel.
        """
        from multiprocessing import Pool, cpu_count
        from functools import partial
        
        # Train separate classifiers in parallel on different segments
        # of the data.
        pool = Pool(processes=None)
        b = 2#cpu_count()
#        print('cpu_count:',b)
        results = pool.map(partial(test_map_train, StreamingExtraTreesClassifier), [(a, b) for a in range(b)])
        print('results:',results)
        pool.close()
        pool.join()
        
        # Now take those classifiers and merge them together to form
        # an approximation of a classifier trained on the entire dataset.
        clf = StreamingExtraTreesClassifier.reduce(*results)
        dataset = load_digits()
        print('trees:',len(clf.trees))
        score = clf.score(dataset.data, dataset.target)
        print('score:',score)
        self.assertEqual(score, 1.0)
    
    def test_map2(self):
        """
        Confirm multiprocessing works even with instance methods.
        """
        from multiprocessing import Pool, cpu_count
        from functools import partial
        
        obj = TestMapTrain()
        pool = Pool(processes=None)
        b = 2
        results = pool.map(partial(obj.train, StreamingExtraTreesClassifier), [(a, b) for a in range(b)])
        print('results:',results)
        pool.close()
        pool.join()
        clf = StreamingExtraTreesClassifier.reduce(*results)
        dataset = load_digits()
        print('trees:',len(clf.trees))
        score = clf.score(dataset.data, dataset.target)
        print('score:',score)
        self.assertEqual(score, 1.0)
        
    def test_joblib(self):
        """
        Confirm joblib's Parallel wrapper around multiprocessing.Pool.map
        is able to train our forest in parallel.
        
        https://pythonhosted.org/joblib/generated/joblib.Parallel.html
        """
        from joblib import Parallel, delayed
        obj = TestMapTrain()
        func = partial(obj.train, StreamingExtraTreesClassifier)
        b = 2
        results = Parallel(n_jobs=-1, verbose=100)(delayed(func)((a, b)) for a in range(b))
        print('results:',results)
        
        clf = StreamingExtraTreesClassifier.reduce(*results)
        dataset = load_digits()
        print('trees:',len(clf.trees))
        score = clf.score(dataset.data, dataset.target)
        print('score:',score)
        self.assertEqual(score, 1.0)
    
if __name__ == '__main__':
    unittest.main()
    