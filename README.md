<img width="1139" alt="image" src="https://user-images.githubusercontent.com/58306690/214125963-d325c142-ab10-4cbc-a60a-862ca911343a.png">


# TS IS FRESH

It is **T**ime **S**eries **I**mportance based **S**election and **F**eatu**R**e **E**xtraction on basis of **S**calable **H**ypothesis tests. 

Algorithm implements the 
idea of combining the selection of features by their importance and the generation of features using the [[tsfresh]](https://github.com/blue-yonder/tsfresh) library :)

Detailed documentation can be found here [**[DOCS]**](https://ts-is-fresh.readthedocs.io/en/latest/index.html#)


## Overview 

**At the first step,** the algorithm tries to understand which features of one time series can be useful.
To do this, it generates a huge number of statistical features using the library. Then he selects them using statistical hypotheses and feature importance values.
![first_stage_ts-is-fresh](https://user-images.githubusercontent.com/58306690/213933487-bb2b0480-cd81-4bd1-add0-1669e35cda35.svg)


**At the second step,** the algorithm uses information about which features were selected from the previous stage. 
For all available time series (target currency and other currencies on the exchange), these features are calculated. 
After that, they also go through two stages of selection - statistical and selection based on importance values.
![second_stage_ts-is-fresh](https://user-images.githubusercontent.com/58306690/213933493-de89a076-dd81-495d-a374-bff49600cd77.svg)

## Results

**First stage.**

Comparison of metrics with constructed (800+) features selected using hypothesis testing and selected using importance values.


|                   |Time (s)|RMSE mean|RMSE l=1|RMSE l=2|RMSE l=3|RMSE l=4|RMSE l=5|RMSE l=6|RMSE l=7|RMSE l=8|RMSE l=9|RMSE l=10|
|-------------------|--------|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|
|all features       |14.4    |0.0183   |0.0145  |0.0156  |0.0169  |0.0169  |0.0175  |0.0185  |0.02    |0.0197  |0.0215  |0.0219   |
|stats selection    |2.6     |0.0195   |0.0168  |0.0168  |0.0186  |0.0204  |0.0202  |0.0202  |0.0202  |0.0202  |0.0208  |0.0212   |
|stats+imp selection|2.2     |0.0186   |0.0158  |0.0161  |0.0176  |0.0186  |0.0188  |0.0193  |0.019   |0.0192  |0.0204  |0.0212   |

We can see that the features selected through statistical tests allow us to reduce training time by a factor of 6, while losing `4.4%` in
accuracy on average across all horizons, and `13%` over the next `300ms`.

With feature importance we reduce the learning time of the model by a factor of `8.3`, losing in accuracy `11.3%` and `17.2%`.


**Second stage.**

Comparison of metrics taking into account contextual information (combining features from the first step for highly correlated time series). 
The metrics after selection are also shown.

|                   |Time (s)|RMSE mean|RMSE l=1|RMSE l=2|RMSE l=3|RMSE l=4|RMSE l=5|RMSE l=6|RMSE l=7|RMSE l=8|RMSE l=9|RMSE l=10|RMSE l=11|RMSE l=12|RMSE l=13|RMSE l=14|RMSE l=15|RMSE l=16|RMSE l=17|RMSE l=18|RMSE l=19|RMSE l=20|
|-------------------|--------|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|only target table  |1.3     |0.118    |0.12    |0.103   |0.1     |0.102   |0.097   |0.116   |0.12    |0.123   |0.124   |0.128    |0.124    |0.128    |0.127    |0.125    |0.123    |0.12     |0.12     |0.119    |0.117    |0.116    |
|with the features of other tables|10.04   |0.096    |0.074   |0.079   |0.081   |0.08    |0.072   |0.103   |0.103   |0.106   |0.103   |0.102    |0.101    |0.106    |0.105    |0.106    |0.103    |0.101    |0.102    |0.101    |0.099    |0.098    |
|with selected features of other tables|1.0     |0.104    |0.136   |0.102   |0.096   |0.099   |0.09    |0.1     |0.101   |0.104   |0.104   |0.103    |0.104    |0.108    |0.11     |0.109    |0.108    |0.105    |0.106    |0.104    |0.102    |0.1      |


In the situation when we leave the features of all correlated currencies, the model learning time increases by `7.7` times, and the quality of the model
improves by `19%`!

After that we want to reduce the learning time of the model by additional selection by `feature_importance`. This selection allows us to reduce the
training time by a factor of `10`, while losing only `8.3%` in accuracy.

Compared to the situation where we only use target currency data, we have an

- `24%` speedup
- `12%` increase in accuracy


## Reproducibility


**requirements.**
```
matplotlib==3.6.3
numba==0.56.4
numpy==1.23.5
pandas==1.5.3
shap==0.41.0
tsfresh==0.20.0
xgboost==1.5.2
```
**data and runs.**

The experiments were run on real data from the stock exchange in one day. The sizes are on the order of several hundred million records.
You can download the dataset from this [[link]](https://drive.google.com/file/d/10cPodvJYP7MEM_6XfAMF99YiDlDxv8wL/view) (and put it into `/data`). Also you can reproduce the results using `/notebooks` (during the execution, the date folder will be filled in as needed).

**file structure.**

```
cd $PATH_TO_PROJECT/docs;
mkdir raw; mkdir separated; mkdir quantized;
```
