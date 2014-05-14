import copy
from collections import defaultdict

from sklearn.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor,
)
from sklearn.ensemble.forest import (
    ForestClassifier, ForestRegressor,
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
)
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

# Allow pickle to serialize instance methods.
import copy_reg
import types

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

class BaseStreamingEstimator(BaseEstimator):
    
    def fit(self, *args, **kwargs):
        return self.partial_fit(*args, **kwargs)
    
    def set_params(self, **params):
        if not params:
            return self
        if 'n_estimators' in params:
            self.n_estimators = params['n_estimators']
            del params['n_estimators']
        self.tree_kwargs.update(params)
    
    def get_params(self, deep=False):
        p = self.tree_kwargs.copy()
        p['n_estimators'] = self.n_estimators
        return p
    
    @classmethod
    def reduce(cls, parts, params={}):
        """
        Merges several classifiers, perhaps trained on separate data subsets,
        into a single classifier.
        
        Note, unless otherwise stated in the params, this assumes that
        parent.n_estimators == parts[0].n_estimators.
        
        You'll likely always want to specify and explicit n_estimators
        for the parent.
        """
        assert parts, 'There must be at least one argument.'
        _params = parts[0].get_params()
        _params.update(params)
        params = _params
        trees = []
        for _ in parts:
            assert isinstance(_, cls), \
                'Can only reduce instances of %s, not %s.' % (cls, type(_))
            trees.extend(copy.deepcopy(_.trees))
#        if 'n_estimators' in params:
#            del params['n_estimators']
        c = cls(**params)
        c.trees = trees
        assert len(c.trees) <= c.n_estimators, \
            "At most %i n_estimators are allowed, but %i trees were found." \
                % (c.n_estimators, len(c.trees))
        return c
    
    @property
    def feature_importances_(self):
        raise NotImplementedError
    
    def fit_transform(self, X, y):
        raise NotImplementedError
    
    def transform(self, X, threshold):
        raise NotImplementedError
    
    def predict_log_proba(self, X):
        raise NotImplementedError

class StreamingForestClassifier(BaseStreamingEstimator, ClassifierMixin):
    
    @property
    def classes_(self):
        return sorted(self.trees[0].classes_)
    
    def set_n_estimators(self, v):
        v = int(v)
        assert v > 0
        self.n_estimators = v
        
    def partial_fit(self, X, y, **kwargs):
#        print('cls:',self)
        
        # Train a new decision tree on the data subset.
        tree = self.base_estimator(**self.tree_kwargs)
        tree.fit(X, y)
        self.trees.append(tree)
        
        # Check relevance of existing trees with current dataset.
        scores = {} # {score:tree}
#        print('trees00:',len(self.trees))
        for tree in self.trees:
            #1.0 is good, 0.0 is bad.
            # We include id() in case a tree has the identical score
            # as another.
            scores[(tree.score(X, y), id(tree))] = tree
        
        # Keep only the N best trees.
        n_estimators = self.n_estimators
#        print('n_estimators0:',n_estimators)
#        print('trees011:',len(self.trees))
#        print('scored:',scores.items())
        ordered_trees = sorted(scores.items(), reverse=True)[:n_estimators]
        self.trees = [_tree for _score, _tree in ordered_trees]
#        print('trees012:',len(self.trees))
    
    def predict(self, X):
        if not self.trees:
            raise Exception("Tree not initialized. Perform a fit first.")
#        label_counts = [defaultdict(int) for _ in xrange(len(X))] # [{label:count}]
#        for tree in self.trees:
#            labels = tree.predict(X) # [label for x at ith position]
##            print 'classes_:',tree.classes_
##            print 'labels:',labels
#            for i, _label in enumerate(labels):
#                label_counts[i][_label] += 1
##        print 'label_counts:',label_counts
#        best_labels = [sorted(label_count.items(), key=lambda o: o[1])[-1][0] for label_count in label_counts]
##        print 'best_labels:',best_labels
#        #best_label, best_count = sorted(counts.items(), key=lambda o: o[1])[-1]
#        return best_labels
        dists = self.predict_proba_dict(X=X)
        ret = []
        for dist in dists:
#            print('dist:',dist)
            best = (-1e99999999, None)
            for label, prob in dist.items():
                best = max(best, (prob, label))
            best_prob, best_label = best
            ret.append(best_label)
        return ret
            
    def predict_proba(self, X):
        if not self.trees:
            raise Exception("Tree not initialized. Perform a fit first.")
        
        def sum_lists(a, b):
            return [_a + _b for _a, _b in zip(a, b)]
        
        # Sum up all distributions from all trees.
        n = len(self.trees[0].classes_)
        dist_counts = [defaultdict(int) for _ in xrange(len(X))] # [{label:prob_sum}]
        for tree in self.trees:
            dists = tree.predict_proba(X) # [[prob of class i, prob of class i+1, etc]]
#            print 'dists:',dists
            #dist_counts = [sum_lists(a, b) for a, b in zip(dist_counts, dist)]
            for i, dist in enumerate(dists):
                for label, prob in zip(tree.classes_, dist):
                    dist_counts[i][label] += prob
#            print 'dist_counts:',dist_counts
        
        # Average the sums.
        m = float(len(self.trees))
        # Normalize class order and value.
        dist_counts = [[label_probs[cls]/m for cls in self.classes_] for label_probs in dist_counts]
#        print 'dist_counts1:',dist_counts
        return dist_counts
    
    def predict_proba_dict(self, *args, **kwargs):
        return [
            dict(zip(self.classes_, dist))
            for dist in self.predict_proba(*args, **kwargs)
        ]

class StreamingForestRegressor(BaseStreamingEstimator, RegressorMixin):
    
    def partial_fit(self, X, y, **kwargs):
        
        # Train a new decision tree on the data subset.
        tree = self.base_estimator(**self.tree_kwargs)
        tree.fit(X, y)
        self.trees.append(tree)
        
        # Check relevance of existing trees with current dataset.
        scores = {} # {score:tree}
        for tree in self.trees:
            #1.0 is good, 0.0 is bad.
            scores[tree.score(X, y)] = tree
        
        # Keep only the N best trees.
        n = self.n_estimators
        self.trees = [
            _tree
            for _score, _tree in sorted(scores.items(), reverse=True)[:n]
        ]
    
    def predict(self, X):
        raise NotImplementedError

class StreamingDecisionTreeClassifier(StreamingForestClassifier):
    """
    A decision tree ensemble that implements partial_fit by training a new
    decision tree for each partial_fit call.
    
    Note, in order for this to work, each partial_fit call should pass
    in the largest amount of data possible.
    """
    
    def __init__(self,
        n_estimators=10,
        criterion='gini',
        splitter='best',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
        min_density=None,
        compute_importances=None):
        
        self.n_estimators = n_estimators
        
        self.trees = []
        self.tree_kwargs = dict(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            min_density=min_density,
            compute_importances=compute_importances,
        )
        
        self.base_estimator = DecisionTreeClassifier

class StreamingDecisionTreeRegressor(StreamingForestRegressor):
    """
    A decision tree ensemble that implements partial_fit by training a new
    decision tree for each partial_fit call.
    
    Note, in order for this to work, each partial_fit call should pass
    in the largest amount of data possible.
    """
    
    def __init__(self,
        n_estimators=10,
        criterion='mse',
        splitter='best',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
        min_density=None,
        compute_importances=None):
        
        self.n_estimators = n_estimators
        
        self.trees = []
        self.tree_kwargs = dict(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            min_density=min_density,
            compute_importances=compute_importances,
        )
        
        self.base_estimator = DecisionTreeRegressor
    
class StreamingRandomForestClassifier(StreamingForestClassifier):
    """
    A decision tree ensemble that implements partial_fit by training a new
    decision tree for each partial_fit call.
    
    Note, in order for this to work, each partial_fit call should pass
    in the largest amount of data possible.
    """
    
    def __init__(self,
        n_estimators=10,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="auto",
        bootstrap=True,
        oob_score=False,
        n_jobs=1,
        random_state=None,
        verbose=0,
        min_density=None,
        compute_importances=None):
        
        self.n_estimators = n_estimators
        
        self.trees = []
        
        self.tree_kwargs = dict(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            min_density=min_density,
            compute_importances=compute_importances,
        )
    
        self.base_estimator = RandomForestClassifier
    
class StreamingRandomForestRegressor(StreamingForestRegressor):
    """
    A decision tree ensemble that implements partial_fit by training a new
    decision tree for each partial_fit call.
    
    Note, in order for this to work, each partial_fit call should pass
    in the largest amount of data possible.
    """
    
    def __init__(self,
        n_estimators=10,
        criterion='mse',
        splitter='best',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
        min_density=None,
        compute_importances=None):
        
        self.n_estimators = n_estimators
        
        self.trees = []
        
        self.tree_kwargs = dict(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            min_density=min_density,
            compute_importances=compute_importances,
        )
        
        self.base_estimator = RandomForestRegressor

class StreamingExtraTreesClassifier(StreamingForestClassifier):
    """
    A decision tree ensemble that implements partial_fit by training a new
    decision tree for each partial_fit call.
    
    Note, in order for this to work, each partial_fit call should pass
    in the largest amount of data possible.
    """
    
    def __init__(self,
        n_estimators=10,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="auto",
        bootstrap=False,
        oob_score=False,
        n_jobs=1,
        random_state=None,
        verbose=0,
        min_density=None,
        compute_importances=None):
        
        self.n_estimators = n_estimators
        
        self.trees = []
        
        self.tree_kwargs = dict(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            min_density=min_density,
            compute_importances=compute_importances,
        )
    
        self.base_estimator = ExtraTreesClassifier
