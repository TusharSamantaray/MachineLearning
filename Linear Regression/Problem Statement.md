## Problem Statement

There are two datasets. The first one has an n x d = 1000 x 50 data matrix (X) called "pred1.dat" with a 1000 x 1 response vector (y) in "resp1.data". The second one has a 1000 x 500 data matrix "pred2.dat" with a response vector in "resp2.dat". These data sets were generated according to the standard linear regression method: y = Xw + e, where X is an n x d matrix of predictor variable, y is an n-dimensional vector of response variable, and $e ~ N(0, \sigma^2 I)$ is an n-dimensional vector of model errors.

##### Part 1 

For each data set, use the first half of the data (observations i = 1; : : : ; n=2, all d predictors) to estimate values for w, ^ w.

##### Part 2

For each data set, use your estimate of w on the 2nd half of the data set (n=2 + 1; :::; n),
to get your estimated response variables, ^y. Compute and report your total squared error:
SSE = Pn i=n=2+1(^yi 􀀀 yi)2

