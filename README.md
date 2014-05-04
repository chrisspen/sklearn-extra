sklearn-extra
=============

Additional classification and regression algorithms based on the scikit-learn library.

Currently, only streaming versions of RandomForestClassifier
and RandomForestRegressor are implemented, called
StreamingRandomForestClassifier and StreamingRandomForestRegressor
respectively. Their use is identical to the prior version, except that these
implement a `partial_fit()`.
