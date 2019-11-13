# Dataset: *Communities and Crime* (Regression 2)

## 1. Total running time: 

about __ minutes


## 2. Results 

### 2.1 Support Vector Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| C                       | 1.0           | 1.0400157     |
| gamma                   | 'auto'        | 0.005033084   |
| kernel                  | 'rbf'         | 'rbf'         |
|                         |               |               |
| **mean_squared_error**  | 0.02535       | 0.02302       |

*(Comment: Due to speed reason, the MSE here calculated only on the first 300 samples)*


### 2.2 Decision Tree Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| max_depth               | None          | 9             |
| min_samples_leaf        | 1             | 36            |
|                         |               |               |
| **mean_squared_error**  | 0.04234       | 0.02249       |


### 2.3 Random Forest Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| n_estimators            | 10            | 273           |
| max_depth               | None          | 35            |
|                         |               |               |
| **mean_squared_error**  | 0.02126       | 0.01914       |


### 2.4 Ada Boost Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| n_estimators            | 50            | 66            |
| learning_rate           | 1.0           | 0.052727222   |
|                         |               |               |
| **mean_sqaured_error**  | 0.02794       | 0.02062       |


### 2.5 Gaussian Process Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| alpha                   | 1e-10         | 0.06650157     |
| kernel                  | 1**2 * RBF(length_scale=1)  | 1**2 * RBF(length_scale=1)  |
|                         |               |               |
| **mean_squared_error**  | 0.10786       | 0.01871       |

*(Comment: huge improvement!)*


### 2.6 Linear Least Squares

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| alpha                   | 1.0           | 108.82026     |
| max_iter                | None          | 1097          |
| solver                  | 'auto'        | 'svd'         |
|                         |               |               |
| **mean_squared_error**  | 0.01955       | 0.01866       |


### 2.7 Neural Network Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| hidden_layer_sizes      | 100           | 538           |
| max_iter                | 200           | 4648          |
|                         |               |               |
| **mean_squared_error**  | 0.06233       | 0.03488       |
