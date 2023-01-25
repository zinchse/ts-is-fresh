.. image:: pictures/ts-is-fresh_logo.png
   :width: 70 %
   :align: center

================================================================================================================
TS IS FRESH
================================================================================================================


Time Series Importance based Selection and FeatuRe Extraction on basis of Scalable Hypothesis tests
---------------------------------------------------------------------------------------------------
This is the documentation of **ts-is-fresh**. Algorithm implements the idea of combining the selection of features by
their importance and the generation of features using the `tsfresh <https://github.com/blue-yonder/tsfresh/>`_ library :)

Project came from the task of building important features for high-frequency trading.

Here is an idea that combines:

- analysis of the similarity of price behavior for different currencies

- automatic generation of statistical statistics by time windows (from ``tsfresh``)

- feature selection through statistical hypothesis tests (from ``tsfresh``)

- reducing the dimension of the feature space due to the clustering of highly correlated features

- counting feature importance values (including ``shap`` values) through block cross validation

- feature selection based on their importance values

Contents
========

.. toctree::
   :maxdepth: 2

   text/introduction
   text/algorithm
   text/toy_example
   text/source_code

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
