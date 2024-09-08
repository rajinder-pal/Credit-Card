Model Documentation
Project Title: Credit Card Fraud Detection

Objective:
The goal of this project is to detect fraudulent credit card transactions from a dataset of real transactions. Using the trained model, we aim to identify patterns that differentiate between genuine and fraudulent activities.

1. Dataset Overview:
Dataset: creditcard.csv
Size: The dataset contains 284,807 transactions, out of which 492 are labeled as fraudulent.
Features: There are 30 features in total, including Time, Amount, and anonymized features (V1 to V28) obtained using PCA.

Class Distribution:
0: Legitimate transactions (majority class)
1: Fraudulent transactions (minority class)


2. Pre-processing:
Handling Imbalanced Data: The dataset has a highly imbalanced class distribution, with fraudulent transactions making up only about 0.17% of the total data.
Normalization:
The features Time and Amount were standardized using StandardScaler to ensure that all features contribute equally to the model.
Data Splitting:
The dataset was split into 70% training and 30% testing using train_test_split from sklearn.


3. Model Selection:
Model Used: RandomForestClassifier
Rationale: Random forests were chosen for their ability to handle both large datasets and imbalanced data, as well as their robustness against overfitting.


4. Model Training:
Feature Engineering: No additional feature engineering was performed as the features were already anonymized.
Model Training:
The model was trained using RandomForestClassifier on the training set.
Hyperparameters were left at their default values, but future tuning could improve accuracy.


5. Evaluation:
Metrics Used:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Accuracy on Test Set: (You can update this based on your results, e.g., 99.95%)


6. Installation Guide: To set up the project locally, follow these steps:

* Clone the repository or unzip the project.

* Navigate to the project directory and install the required dependencies:


Copy code
pip install -r requirements.txt
Ensure Python 3.x is installed on your machine.

Install required Python libraries:

bash
Copy code
pip install Flask
pip install pandas
pip install scikit-learn
pip install numpy
pip install matplotlib
pip install seaborn


7. How to Run the Application:

In the terminal or command prompt, navigate to the project directory.

Run the Flask application using the command:

bash
Copy code
python app.py
Once the app is running, open a web browser and visit http://127.0.0.1:5000/ to interact with the model.

8. Limitations:
Class Imbalance: The dataset is highly imbalanced. Even with a high accuracy score, the model might have a higher false negative rate, which could be problematic for detecting fraudulent cases.
Overfitting Risk: Since RandomForest can be prone to overfitting, the model may need hyperparameter tuning to generalize better.