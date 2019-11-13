# Dataset: *Default of credit card clients* (Classification 2)

## 1. Total running time: 

around 40 minutes


## 2. Results 

### 2.1 K-Nearest Neighbours Classifier

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| n_neighbors   | 5             | 36            |
| weights       | 'uniform'     | 'distance'    |
|               |               |               |
| **accuracy**  | 79.64 %       | 81.32 %       |


### 2.2 Support Vector Classifier

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| C             | 1.0           |       ?       |
| gamma         | 'auto'        |       ?       |
| kernel        | 'rbf'         |       ?       |
|               |               |       ?       |
| **accuracy**  | 82.48 %       |       ?       |

*(Comment: svc didn't finish in 1 hour, because it is very very slow to train on all training set. However, the original model is not bad among all models w.r.t. accuracy. We decide to use the raw model)*


### 2.3 Decision Tree Classifier

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| criterion     | 'gini'        | 'gini'        |
| max_depth     | None          | 4             |
|               |               |               |
| **accuracy**  | 72.67 %       | 82.54 %       |


### 2.4 Random Forest Classifier

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| criterion     | 'gini'        | 'entropy'        |
| n_estimators  | 100           | 36            |
| max_depth     | None          | 9            |
|               |               |               |
| **accuracy**  | 80.90 %       | 82.51 %       |


### 2.5 Ada Boost Classifier

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| n_estimators  | 50            | 24            |
| learning_rate | 1.0           | 0.00010978738 |
|               |               |               |
| **accuracy**  | 82.16 %       | 82.62 %       |


### 2.6 Logistic Regression

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| C             | 1.0           | 2.70857856061711 |
| max_iter      | 100           | 797           |
|               |               |               |
| **accuracy**  | 81.65 %       | 81.65 %       |

*(Comment: same accuracy)*


### 2.7 Gaussian Naïve Bayes

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| var_smoothing | 1e-09         | 0.0027052340310675443 |
|               |               |               |
| **accuracy**  | 59.32 %       | 59.67 %       |

*(Comment: no much improve)*


### 2.8 Multi-Layer Perceptron Classifier

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| hidden_layer_sizes | 100      | 285           |
| max_iter      | 200           | 8018          |
|               |               |               |
| **accuracy**  | 81.14 %       | 78.99 %       |

*(Comment 1: MLP Classifier finished in around 15 minutes and got lower accuracy)*

*(Comment 2: Due to the reason that the tuned parameters' accuracy is much lower, I choose the use the original parameters)*

