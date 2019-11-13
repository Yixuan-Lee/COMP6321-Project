# Dataset: *Diabetic Retinopathy* (Classification 1)

## 1. Total running time: 

10-20 minutes


## 2. Results 

### 2.1 K-Nearest Neighbours Classifier

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| n_neighbors   | 5             | 69            |
| weights       | 'uniform'     | 'distance'    |
|               |               |               |
| **accuracy**  | 63.95 %       | 65.79 %       |

*(Comment: actually n_neighbors=74 has the best accuracy (66.32 %) on actual test set)* 


### 2.2 Support Vector Classifier

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| C             | 1.0           | 114.05387687591531     |
| gamma         | 'auto'        | 0.0011408824551981213  |
| kernel        | 'rbf'         | 'linear'      |
|               |               |               |
| **accuracy**  | 66.84 %       | 74.47 %       |


### 2.3 Decision Tree Classifier

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| criterion     | 'gini'        | 'entropy'     |
| max_depth     | None          | 7             |
|               |               |               |
| **accuracy**  | 64.47 %       | 58.68 %       |

*(Comment: Due to the reason that the tuned parameters' accuracy is much lower, I choose the use the original parameters)*


### 2.4 Random Forest Classifier

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| criterion     | 'gini'        | 'gini'        |
| n_estimators  | 100           | 33            |
| max_depth     | None          | 29            |
|               |               |               |
| **accuracy**  | 65.79 %       | 69.21 %       |

*(Comment: actually n_estimator=32, max_depth=32 has a little better result of accuracy of 69.47 % on actual test set)*


### 2.5 Ada Boost Classifier

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| n_estimators  | 50            | 79            |
| learning_rate | 1.0           | 0.9483906     |
|               |               |               |
| **accuracy**  | 66.84 %       | 67.63 %       |


### 2.6 Logistic Regression

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| C             | 1.0           | 120.00786     |
| max_iter      | 100           | 926           |
|               |               |               |
| **accuracy**  | 73.68 %       | 74.47 %       |


### 2.7 Gaussian Naïve Bayes

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| var_smoothing | 1e-09         | 1.4001572e-08 |
|               |               |               |
| **accuracy**  | 61.84 %       | 64.47 %       |


### 2.8 Multi-Layer Perceptron Classifier

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| hidden_layer_sizes | 100      | 923           |
| max_iter      | 200           | 1159          |
|               |               |               |
| **accuracy**  | 72.89 %       | 75.26 %       |

*(Comment: hidden_layer_sizes=919, max_iter=1047 has a little better result of accuracy of 76.58 % on actual test set)*

