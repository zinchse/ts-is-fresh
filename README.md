<p align="center">
  <img width="300" alt="image" src="https://user-images.githubusercontent.com/58306690/214125963-d325c142-ab10-4cbc-a60a-862ca911343a.png">
</p>

# 🍹 TS IS FRESH

**T**ime **S**eries **I**mportance based **S**election and **F**eatu**R**e **E**xtraction on basis of **S**calable **H**ypothesis tests. 

The algorithm incorporates a combination of feature selection based on their
importance and feature generation using the [[tsfresh]](https://github.com/blue-yonder/tsfresh) lib.
By adopting this approach, I achieved on real data from the stock exchange the increase in prediction 
accuracy by `12%`, while improving model performance with a speedup of `24%`.

## 🗺️ Overview 

**At the first step,** the algorithm tries to understand which features of one time series can be useful.
To do this, it generates a huge number of statistical features using the ``tsfresh`` library. 
Then it selects them using statistical hypotheses and feature importance values.
All values are calculated using block Cross-Validation schema.

<p align="center">
  <img width="600" alt="image" src="https://user-images.githubusercontent.com/58306690/213933487-bb2b0480-cd81-4bd1-add0-1669e35cda35.svg">
</p>




**At the second step,** the algorithm uses information about which features were selected from the previous stage. 
For all **correlated** and available time series (other currencies on the exchange), these features are also calculated. 
After that, they also go through two stages of selection - statistical and selection based on importance values.

<p align="center">
  <img width="600" alt="image" src="https://user-images.githubusercontent.com/58306690/213933493-de89a076-dd81-495d-a374-bff49600cd77.svg">
</p>


## 📊 Results

Compared to the situation where we only use target currency data, we have the `24%` speedup **and** `12%` increase in accuracy!

|                   |Time (s)|RMSE (mean)|
|-------------------|--------|---------|
|only target table  |1.3     |0.118    |
|with the features of other tables|10.04   |0.096    |
|with selected features of other tables|1.0     |0.104    |


## 🚀 Quick Start

The [[dataset]](https://drive.google.com/file/d/10cPodvJYP7MEM_6XfAMF99YiDlDxv8wL/view) size has order of several hundred million records.
To reproduce my result You can extract it in `data/raw` folder and use .ipynb from `/notebooks`.
