# Dataset: *Dataset Title* (Regression __)

## 1. Total running time: 

about __ minutes


## 2. Results 

### 2.1 Support Vector Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| C                       | 1.0           |               |
| gamma                   | 'auto'        |               |
| kernel                  | 'rbf'         |               |
|                         |               |               |
| **mean_sqaured_error**  |               |               |
| **r2_score**            |               |               |


### 2.2 Decision Tree Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| max_depth               | None          |               |
| min_samples_leaf        | 1             |               |
|                         |               |               |
| **mean_sqaured_error**  |               |               |
| **r2_score**            |               |               |


### 2.3 Random Forest Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| n_estimators            | 10            |               |
| max_depth               | None          |               |
|                         |               |               |
| **mean_sqaured_error**  |               |               |
| **r2_score**            |               |               |


### 2.4 Ada Boost Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| n_estimators            | 50            |               |
| learning_rate           | 1.0           |               |
|                         |               |               |
| **mean_sqaured_error**  |               |               |
| **r2_score**            |               |               |


### 2.5 Gaussian Process Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| alpha                   | 1e-10         |      |
| kernel                  | 1**2 * RBF(length_scale=1)  |      |
|                         |               |               |
| **mean_sqaured_error**  |               |               |
| **r2_score**            |               |               |


### 2.6 Linear Least Squares

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| alpha                   | 1.0           |               |
| max_iter                | None          |               |
| solver                  | 'auto'        |               |
|                         |               |               |
| **mean_sqaured_error**  |               |               |
| **r2_score**            |               |               |


### 2.7 Neural Network Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| hidden_layer_sizes      | 100           |               |
| max_iter                | 200           |               |
|                         |               |               |
| **mean_sqaured_error**  |               |               |
| **r2_score**            |               |               |

