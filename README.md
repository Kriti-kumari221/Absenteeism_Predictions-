📌 Absenteeism Prediction Project

This project builds and deploys a machine learning model to predict whether an employee is likely to be absent from work.
The workflow includes data preprocessing, feature engineering, custom scaling, training a Logistic Regression model, and using the model to predict new data.

🚀 Project Workflow
1. Data Preprocessing

Raw absenteeism dataset is cleaned:

Dropped unnecessary columns (ID, Absenteeism Time in Hours during prediction).

Converted categorical Reason for Absence into dummy variables and grouped into 4 types (Reason_1 … Reason_4).

Extracted date-related features:

Month Value

Day of the Week

Mapped Education into binary (0 = high school, 1 = higher education).

Handled missing values (replaced with 0).

Dropped redundant features (Day of the Week, Distance to Work, Daily Work Load Average).

2. Feature Scaling

Implemented a custom scaler CustomScaler class:

Standardizes selected numerical features.

Keeps feature order consistent.

Columns scaled:

['Month Value', 'Transportation Expense', 'Distance to Work', 
 'Age', 'Daily Work Load Average', 'Body Mass Index', 
 'Education', 'Children', 'Pet']

3. Model Training

Logistic Regression was trained on the preprocessed & scaled data.

Balanced for class imbalance using class_weight="balanced" if needed.

Saved the model and scaler as .pkl files using pickle.

4. Absenteeism Model Class

Defined in absenteeism_module.py:

load_and_clean_data(file) → preprocess new CSV input.

predicted_probability() → gives probability of being absent.

predicted_output_category() → returns 0/1 prediction.

predicted_outputs() → returns dataframe with probabilities and predictions added.

5. Making Predictions

Example:

from absenteeism_module import absenteeism_model

# Load model and scaler
model = absenteeism_model('model', 'scaler')

# Load new data and preprocess
model.load_and_clean_data('Absenteeism_new_data.csv')

# Get predictions
results = model.predicted_outputs()
print(results[['Probability', 'Prediction']])

📂 Project Structure
.
├── absenteeism_module.py      # main model class
├── absenteeism_model.pkl      # saved logistic regression model
├── scaler.pkl                 # saved custom scaler
├── Absenteeism_data.csv       # training dataset
├── Absenteeism_new_data.csv   # new input data for testing
├── notebooks/                 # Jupyter notebooks for training & exploration
└── README.md                  # project documentation

⚠️ Common Issues & Fixes

KeyError: 'Pets' not in index → Check column naming (Pet vs Pets) and keep it consistent.

UserWarning: X does not have valid feature names → Ensure you pass a DataFrame with correct column names, not a raw NumPy array.

Always predicting 1 → Might be class imbalance or input not preprocessed the same way as training.

✅ Requirements

Python 3.8+

pandas

numpy

scikit-learn

Install dependencies:

pip install -r requirements.txt

📊 Output

The final predictions file will contain:

All input features (after preprocessing).

A Probability column (likelihood of absence).

A Prediction column (0 = not absent, 1 = absent).

✨ With this project, you can predict employee absenteeism on unseen datasets using a robust preprocessing pipeline and logistic regression model.
