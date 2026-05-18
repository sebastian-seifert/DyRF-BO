# Journal for Research Progress

## 18.5
Today i expanded the comparison of the different epistemic uncertainty quantification approaches. (Standard, Chen, Shaker)
I started approximating the integral from Shaker using Adaptive Quadrature, and getting all of the tree variances by just computing the variances from the training set. This could be further improved. 
For evaluation, i use a sin-wave on the interval from [0,10] with some gaussian noise added on top.
The random forest is then trained on [0,10] \ [4,6], using the OOD technique. 
I then measure the NLL value, the correlation between MSE and predicted total variance (Epi + expected value of the Tree Variances (Aleatoric)), as well as the ratio between gap and non-gap variance. 
The NLL for everything but Shaker is realized via a simple "Gaussian Plug In", the GMM from Shaker is more complicated to compute. After running some tests, the NLL value of Shaker seems to be the best.

ToDo: Run statistical tests to prove my intuition. 
