# Artificial Neural Network for Paris Housing Price Prediction

## Project Overview

This project focuses on predicting housing prices in Paris using an Artificial Neural Network (ANN). The dataset includes various features of housing properties such as size, number of rooms, and amenities, and the goal is to predict the price of a property based on these characteristics.

## Dataset

The dataset contains over 10,000 records of housing properties in Paris with features like:
- `squareMeters`, `numberOfRooms`, `hasYard`, `hasPool`, `made` (year built), and more.
- `price` (target variable) represents the sale price of the property.

## Model Approach

The project uses an ANN for regression tasks. The ANN model captures complex patterns in the dataset, allowing it to predict housing prices based on the provided features.

## Data Preprocessing

Key preprocessing steps:
- Handling missing values in the `numberOfRooms` column.
- Removing irrelevant columns like `numPrevOwners`.
- Shuffling the dataset to ensure randomness in training.

## Model Architecture

A simple feedforward ANN with three hidden layers was chosen. Experiments with different configurations were conducted to find the optimal model architecture based on RMSE performance.

## Results

The model's RMSE values were tested under different configurations. The best-performing models demonstrated an RMSE of approximately 3,795.3, suggesting a solid prediction accuracy.

## Conclusion

This project demonstrates how an ANN can be used for predicting housing prices, providing a valuable tool for understanding market trends in Paris.
