# Dataset: *Wine Quality* (Regression 1)

## 1. Total running time: 

about __ minutes


## 2. Results 

### 2.1 Support Vector Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| C                       | 1.0           | 1.0122664227574725  |
| gamma                   | 'auto'        | 0.15143539905722755 |
| kernel                  | 'rbf'         | 'rbf'         |
|                         |               |               |
| **mean_sqaured_error**  | 0.38955       | 0.38558       |


### 2.2 Decision Tree Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| max_depth               | None          | 5             |
| min_samples_leaf        | 1             | 7             |
|                         |               |               |
| **mean_sqaured_error**  | 0.71515       | 0.46920       |


### 2.3 Random Forest Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| n_estimators            | 10            | 1200          |
| max_depth               | None          | 54            |
|                         |               |               |
| **mean_sqaured_error**  | 0.41997       | 0.36808       |


### 2.4 Ada Boost Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| n_estimators            | 50            | 24            |
| learning_rate           | 1.0           | 1.1344235     |
|                         |               |               |
| **mean_sqaured_error**  | 0.41356       | 0.41845       |

*(Comment: worse results)*


### 2.5 Gaussian Process Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| alpha                   | 1e-10         | 0.5440077     |
| kernel                  | 1**2 * RBF(length_scale=1)  | 1**2 * RBF(length_scale=0.5)              |
|                         |               |               |
| **mean_sqaured_error**  | 3.17726       | 0.42887       |

*(Comment: huge improvement!)*


### 2.6 Linear Least Squares

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| alpha                   | 1.0           | 13.528105     |
| max_iter                | None          | 144           |
| solver                  | 'auto'        | 'saga'        |
|                         |               |               |
| **mean_sqaured_error**  | 0.44277       | 0.44089       |


### 2.7 Neural Network Regressor

|       Parameters        |       Raw     |     Tuned     |
| ----------------------- | ------------- | ------------- |
| hidden_layer_sizes      | 100           | 814           |
| max_iter                | 200           | 1270          |
|                         |               |               |
| **mean_sqaured_error**  | 0.73434       | 0.45231       |

