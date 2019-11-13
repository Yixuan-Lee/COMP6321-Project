# Dataset: *Seismic Bumps* (Classification 10)

## 1. Total running time: 

about _____ minutes


## 2. Results 

### 2.1 K-Nearest Neighbours Classifier

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| n_neighbors   | 5             | 46            |
| weights       | 'uniform'     | 'distance'    |
|               |               |               |
| **accuracy**  | 93.90 %       | 94.37 %       |


### 2.2 Support Vector Classifier

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| C             | 1.0           | 1.0743160846562865  |
| gamma         | 'auto'        | 0.00014283093768731805  |
| kernel        | 'rbf'         | 'linear'      |
|               |               |               |
| **accuracy**  | 94.37 %       | 94.37 %       |


### 2.3 Decision Tree Classifier

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| criterion     | 'gini'        | 'gini'        |
| max_depth     | None          | 1             |
|               |               |               |
| **accuracy**  | 86.64 %       | 94.37 %       |


### 2.4 Random Forest Classifier

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| criterion     | 'gini'        | 'entropy'     |
| n_estimators  | 100           | 19            |
| max_depth     | None          | 24            |
|               |               |               |
| **accuracy**  | 93.55 %       | 93.08 %       |

*(Comment: raw model performs better. We decide to use the raw model)*


### 2.5 Ada Boost Classifier

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| n_estimators  | 50            | 1             |
| learning_rate | 1.0           | 0.00027640525 |
|               |               |               |
| **accuracy**  | 94.02 %       | 94.37 %       |


### 2.6 Logistic Regression

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| C             | 1.0           | 2.7640524e-06 |
| max_iter      | 100           | 20            |
|               |               |               |
| **accuracy**  | 94.26 %       | 94.37 %       |


### 2.7 Gaussian Naïve Bayes

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| var_smoothing | 1e-09         | 135.28105     |
|               |               |               |
| **accuracy**  | 83.47 %       | 94.37 %       |


### 2.8 Multi-Layer Perceptron Classifier

|   Parameters  |       Raw     |     Tuned     |
| ------------- | ------------- | ------------- |
| hidden_layer_sizes | 100      | 446           |
| max_iter      | 200           | 1334          |
|               |               |               |
| **accuracy**  | 94.26 %       | 92.26 %       |

*(Comment: worse result after tuning, so we decide to use the raw model)*
