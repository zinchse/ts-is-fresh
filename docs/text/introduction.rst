Introduction
============


Overview
--------

Hereinafter, by **ts-is-fresh** we will mean the idea of combining ``tsfresh``, ``block cross validation``, and
``feature importance selection``.

The ``tsfresh`` library **ts-is-fresh** combines the automatic search for significant features, which is very important
for high-frequency trading. In a situation where a huge number of trades occur every second, and with the rapidly
changing market, it is impossible to "hand-assess" the situation. It is necessary to build systems, which are able to
select the important information and use it for increasing the accuracy of forecasts. Since ``tsfresh`` has a huge
number of attributes and many of them take quite a long time to calculate, an additional selection of functions from
``tsfresh`` has been implemented to be used within the **ts-is-fresh** algorithm.


Also, thanks to the block cross validation **ts-is-fresh** pays attention not only to the latest changes in the time
series, but also to the market behavior over the whole time range. Broadly speaking, block cross validation evenly
divides the whole time series into blocks, at each block different statistics (``p_values``, ``feature_importance``,
metrics, etc.) are counted, and then these indicators are averaged. Thanks to this technique, we do not focus our
forecasting only on the last values of the time series. The block cross validation scheme takes into account the entire
structure of the time series, does not change the sequence of events, and avoids data leaks.

To learn more about how block cross-validation works, see :ref:`Algorithm`.


What problem does **ts-is-fresh** solve?
----------------------------------------

**ts-is-fresh** is built to construct new features for predicting cryptocurrency prices on exchanges.
Because of the high frequency of trading in this area, the built solution must work **very quickly** and **not require**
**manual debugging**. For this reason it is necessary not only to build additional features, which will help increase
the accuracy of the predictions, but also to **limit their size**! We can't afford a long inference of models,
nor a long learning process.


How does it solve this problem?
-------------------------------

It was decided to generate a large number of statistical features, then train a gradient-boosting model and, using
feature counting, leave only the most important features. From time to time it will be necessary to train the models
on a large number of features to understand which subset of the features is now the most useful. But once we have
selected the most useful features, we can train the models for a long time on only that set of features. Because we
chose ``XGBR`` model, we have the ability to select  by feature importance values as well.


And why exactly in this way?
----------------------------

Let's understand what the solutions are in general:

**A) smart feature engineering**: using domain knowledge, important features are created by hand, over which a simple
(e.g., linear) model is then built

* easy to further train on-line
* it's interpretable
* very fast model inference
* domain knowledge is needed

**B) semi-automatic feature engineering**: using some heuristics, different kinds of statistics (medians, quantiles,
etc.) are computed, over which then treebased models are built

* less demanding of domain knowledge (because of the use of a more complex model, we can afford to build less expressive features)
* high expressive power
* fast model inference
* cannot be quickly retrained on-line
* it's uninterpretable

**C) statistical autoregressive approach**: models like Arima, Prophet, etc.

* fast model inference
* correct selection of hyperparameters is necessary to build a good model

**D) RNN-like approaches:** recurrent neural networks like LSTM and others

* very heavy models (in terms of training and inference)
* can show very good results

Due to my limited knowledge of the cryptocurrency market, I am removing the **A)** option. Since we have a lot of data,
it will be quite hard to train high quality statistical models (to enumerate hyperparameters). Because of this approach
**C)** is also rejected. Our goal is to predict ``300ms`` ahead, because of the fact that in approach **D)** this is
comparable to inference models, it is also removed.

This leaves approach **B)**, in which we need to automatically construct good features. Moreover, because of the
limitation on inference and the lack of on-line retraining, our model must work fast enough (there must not be very
many features), and also have a prediction horizon comparable to the learning time of the new model (we must have a
good model at every moment, if the model is built longer than its predictions become obsolete, we will not be able
to trade).

