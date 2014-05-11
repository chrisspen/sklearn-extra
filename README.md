sklearn-extra
=============

Additional classification and regression algorithms based on the
[scikit-learn library](scikit-learn.org).

Currently implemented classes are:

* StreamingDecisionTreeClassifier
* StreamingRandomForestClassifier
* StreamingExtraTreesClassifier

Their use and parameters are essentially identical to their non-streaming
originals, except that these implement partial_fit().

Note, that for datasets that can be trained in-memory with a single batch,
these will perform no better than the originals. Their only benefit lies when
training on a dataset too large to fit into memory, requiring the dataset to
be split into smaller parts.