# Zoo Animal Classification Project
## Project Overview

This project focuses on classifying zoo animals into 7 categories based on 16 features, including the presence of hair, feathers, eggs, milk secretion, etc. The dataset consists of 101 records, each representing an animal with these features and its corresponding class label. The goal is to explore and apply different machine learning models to predict the correct class of each animal.

## Models Used

- **Perceptron**: A simple neural network model used to classify animals based on feature weights and learning rate adjustments.
- **Naive Bayes**: A probabilistic classifier based on the assumption of feature independence, with variants like Gaussian and Multinomial.
- **K-Nearest Neighbors (KNN)**: A non-parametric model that classifies animals by calculating distances to the nearest neighbors.
- **Support Vector Machine (SVM)**: A classifier that separates classes with the largest margin, ideal for complex classification tasks.
- **Decision Trees**: A model that splits the data based on feature values to classify animals, with clear decision-making paths.

## Key Features

- Cross-validation to evaluate model performance.
- Hyperparameter tuning for optimal results.
- Performance evaluation based on accuracy.

## Dataset

The dataset used for this project is the "Zoo Animal Classification" dataset from Kaggle, containing 101 records and 16 features. The target variable is the `class_type`, which categorizes animals into one of the following classes:  
- Amphibian  
- Bird  
- Bug  
- Fish  
- Invertebrate  
- Mammal  
- Reptile  

## Results

The models performed well on the dataset, with **Naive Bayes (Gaussian)** achieving the highest accuracy of **95%**. The performance of other models like **KNN**, **Perceptron**, and **SVM** also showed strong results, with accuracies ranging from **85% to 95%**.

## Conclusion

The project demonstrates the application of several machine learning techniques for classification tasks. It provides insights into model performance and efficiency based on the dataset's features. The results indicate that simpler models like **Naive Bayes** work very well, while more complex models like **SVM** and **Decision Trees** also perform effectively.

---

For more details, check out the full [documentation](/Doc.pdf) and code within the repository.
