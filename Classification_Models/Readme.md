# Game of Thrones Character Survival Prediction

This project builds and evaluates machine learning models to predict whether a character from the *Game of Thrones* series is alive or dead. Using features like gender, nobility status, and book appearances, the models classify characters' survival status with up to 73.5% accuracy.

## Dataset

The dataset was sourced from Kaggle and includes information about characters such as:

- Gender (male/female)
- Nobility status (noble or commoner)
- Number of books each character appears in
- Survival status (alive or dead)

## Methodology

1. **Data Preprocessing**
   - Handling missing data for gender and nobility
   - Encoding categorical variables
   - Splitting data into training (70%) and testing (30%) sets

2. **Models Used**
   - Logistic Regression
   - Random Forest Classifier

3. **Evaluation**
   - Models were compared based on accuracy, precision, recall, and F1-score
   - Both models reached approximately 73.5% accuracy

4. **Model Interpretation**
   - LIME (Local Interpretable Model-agnostic Explanations) was used to explain predictions
   - Nobility and book appearances were key predictive features

## Results

- Logistic Regression showed balanced sensitivity and specificity but was weaker in predicting deaths.
- Random Forest had better precision and recall for alive characters and balanced class predictions.
- Both models highlight nobility and book presence as important factors for survival.

## Future Work

- Incorporate additional features such as house affiliation, alliances, or age.
- Explore other machine learning models for better accuracy.
- Perform feature engineering to enhance prediction quality and interpretability.
