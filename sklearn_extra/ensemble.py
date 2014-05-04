from collections import defaultdict

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

class BaseStreamingRandomForestEstimator(BaseEstimator):
    
    def set_params(self, **params):
        if not params:
            return self
        if 'n_estimators' in params:
            self.n_estimators = params['n_estimators']
            del params['n_estimators']
        self.tree_kwargs.update(params)
    
    def get_params(self, deep):
        p = self.tree_kwargs.copy()
        p['n_estimators'] = self.n_estimators
        return p

class StreamingRandomForestClassifier(BaseStreamingRandomForestEstimator, ClassifierMixin):
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
    
    def fit(self, *args, **kwargs):
        return self.partial_fit(*args, **kwargs)
    
    def partial_fit(self, X, y, **kwargs):
        
        # Train a new decision tree on the data subset.
        tree = DecisionTreeClassifier(**self.tree_kwargs)
        tree.fit(X, y)
        self.trees.append(tree)
        
        # Check relevance of existing trees with current dataset.
        scores = {} # {score:tree}
        for tree in self.trees:
            #1.0 is good, 0.0 is bad.
            scores[tree.score(X, y)] = tree
        
        # Keep only the N best trees.
        n_estimators = self.n_estimators
        self.trees = [_tree for _score, _tree in sorted(scores.items(), reverse=True)[:n_estimators]]
    
    @property
    def classes_(self):
        return sorted(self.trees[0].classes_)
    
    def predict(self, X):
        if not self.trees:
            raise Exception("Tree not initialized. Perform a fit first.")
        label_counts = [defaultdict(int) for _ in xrange(len(X))] # [{label:count}]
        for tree in self.trees:
            labels = tree.predict(X) # [label for x at ith position]
#            print 'classes_:',tree.classes_
#            print 'labels:',labels
            for i, _label in enumerate(labels):
                label_counts[i][_label] += 1
#        print 'label_counts:',label_counts
        best_labels = [sorted(label_count.items(), key=lambda o: o[1])[-1][0] for label_count in label_counts]
#        print 'best_labels:',best_labels
        #best_label, best_count = sorted(counts.items(), key=lambda o: o[1])[-1]
        return best_labels
            
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
    
    @property
    def feature_importances_(self):
        raise NotImplementedError
    
    def fit_transform(self, X, y):
        raise NotImplementedError
    
    def transform(self, X, threshold):
        raise NotImplementedError
    
    def predict_log_proba(self, X):
        raise NotImplementedError

class StreamingRandomForestRegressor(BaseStreamingRandomForestEstimator, RegressorMixin):
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
    
    def fit(self, *args, **kwargs):
        return self.partial_fit(*args, **kwargs)
    
    def partial_fit(self, X, y, **kwargs):
        
        # Train a new decision tree on the data subset.
        tree = DecisionTreeRegressor(**self.tree_kwargs)
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
        if not self.trees:
            raise Exception("Tree not initialized. Perform a fit first.")
        label_counts = [defaultdict(int) for _ in xrange(len(X))] # [{label:count}]
        #TODO
        for tree in self.trees:
            labels = tree.predict(X) # [label for x at ith position]
            print 'labels:',labels
            for i, _label in enumerate(labels):
                label_counts[i][_label] += 1
#        print 'label_counts:',label_counts
        todo
        best_labels = [sorted(label_count.items(), key=lambda o: o[1])[-1][0] for label_count in label_counts]
#        print 'best_labels:',best_labels
        #best_label, best_count = sorted(counts.items(), key=lambda o: o[1])[-1]
        return best_labels
    
    @property
    def feature_importances_(self):
        raise NotImplementedError
    
    def fit_transform(self, X, y):
        raise NotImplementedError
    
    def transform(self, X, threshold):
        raise NotImplementedError
    