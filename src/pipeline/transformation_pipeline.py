"""
This module contains a class for data  transformation, and feature engineering pipelines. It is designed to handle the preprocessing of data for machine learning tasks, focusing on categorical transformation, normalization, standardization, and feature engineering. The module includes the following key components:

- LazyTransformationPipeline: A class that orchestrates the data transformation pipeline using Polars' lazy API. This class handles tasks such as categorical encoding, splitting data into train_data/test_data sets, applying normalization or standardization, and performing feature engineering.

The pipeline supports various machine learning models, including SVM, KNN, PCA, and tree-based models, and provides mechanisms for hyperparameter tuning using RandomSearch, GridSearch, or Bayesian Optimization.

Modules used:
- Polars for efficient data manipulation and transformation
- Scikit-learn for hyperparameter tuning
- Skopt for Bayesian optimization
"""
