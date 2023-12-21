# Diabetes Prediction

This Python code is a machine learning script that uses a Support Vector Machine (SVM) classifier to predict the outcome of diabetes based on a given dataset. Here's a breakdown of the code:

1. **Import Libraries:**
   - `numpy` and `pandas` for data manipulation.
   - `StandardScaler` from `sklearn.preprocessing` for feature scaling.
   - `train_test_split` from `sklearn.model_selection` for splitting the dataset into training and testing sets.
   - `accuracy_score` from `sklearn.metrics` to evaluate the accuracy of the model.
   - `svm` from `sklearn` for Support Vector Machine classifier.

2. **Load and Explore Data:**
   - The code reads a dataset named 'diabetes.csv' using Pandas.
   - The first five rows of the dataset are displayed using `head()`.

3. **Data Preprocessing:**
   - The features (X) are separated from the target variable (Y).
   - The features are scaled using `StandardScaler` to standardize them.

4. **Train-Test Split:**
   - The dataset is split into training and testing sets using `train_test_split`.
   - The split is stratified based on the target variable 'Outcome'.

5. **SVM Model Training:**
   - A linear Support Vector Machine (SVM) classifier is created.
   - The classifier is trained on the training data.

6. **Model Evaluation - Training Set:**
   - The model's predictions are calculated for the training set.
   - The accuracy of the model on the training set is computed using `accuracy_score`.

7. **Model Evaluation - Testing Set:**
   - The model's predictions are calculated for the testing set.
   - The accuracy of the model on the testing set is computed using `accuracy_score`.

8. **Make Predictions on New Data:**
   - Six sets of input data are provided for predicting diabetes outcomes.
   - The input data is converted to a NumPy array.
   - The trained classifier is used to predict the outcomes for the new input data.

9. **Print Predictions:**
   - The predicted outcomes for the new data are printed.
   - The unique values and their counts in the predictions are displayed.

This script essentially loads a diabetes dataset, preprocesses it, trains a linear SVM classifier, evaluates its accuracy on both training and testing sets, and makes predictions on new data. The SVM model's performance is assessed using accuracy metrics.
