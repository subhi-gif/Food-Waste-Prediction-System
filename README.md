# 🍽️ Food Waste Prediction

This is the ML project which predicts food wastage using machine learning based on kitchen, operational, environmental, and historical waste factors.

The main objective of this project is to predict the expected amount of food waste in kilograms and provide useful insights that can help kitchens make better food preparation and waste-management decisions.

The project compares multiple machine learning regression models using MAE, MSE, RMSE, and R² Score. After model comparison, time-based evaluation, and hyperparameter tuning, Linear Regression was selected as the final deployment model.

The final model is integrated into an interactive Streamlit web application that allows users to enter kitchen and operational information and receive a food-waste prediction, waste-risk level, and recommendation.

---

## 🎯 Project Objective

Food waste is a major issue in restaurants, cafeterias, institutional kitchens, and other food-service environments. Preparing more food than required can lead to unnecessary leftovers and increased food wastage.

This project aims to develop a machine learning system that can:

- Predict expected food waste in kilograms
- Analyze factors affecting food waste
- Compare different machine learning models
- Evaluate model performance using regression metrics
- Use historical and operational information to predict future food waste
- Provide a waste-risk level
- Provide recommendations based on predicted waste
- Provide an interactive Streamlit interface

---

## 🧠 Problem Type

This project is a **Machine Learning Regression Problem**.

The model predicts a continuous numerical value.

**Target Variable:** `food_waste_kg`

The output represents the predicted amount of food waste in kilograms.

> **Note:** Since this is a regression problem, R² Score is used as a model-performance metric rather than classification accuracy.

---

## 📊 Dataset and Features

The dataset contains information related to kitchen operations, environmental conditions, staff experience, waste categories, and previous food waste.

### Input Features

| Feature | Description |
|---|---|
| `meals_served` | Number of meals served |
| `kitchen_staff` | Number of kitchen staff |
| `temperature_C` | Temperature in Celsius |
| `humidity_percent` | Humidity percentage |
| `day_of_week` | Day of the week |
| `special_event` | Whether a special event occurred |
| `past_waste_kg` | Previous food waste in kilograms |
| `staff_experience` | Experience level of kitchen staff |
| `waste_category` | Category of food waste |
| `year` | Year extracted from date |
| `month` | Month extracted from date |
| `day` | Day extracted from date |
| `week_of_year` | Week number of the year |
| `is_weekend` | Indicates whether the date falls on a weekend |

### Target Variable

`food_waste_kg`

---

## 🧹 Data Preprocessing

Before training the models, the dataset was cleaned and prepared for machine learning.

The preprocessing process included:

- Data cleaning
- Data type conversion
- Date conversion
- Handling categorical variables
- Feature engineering
- Separation of input features and target variable
- Encoding categorical features
- Scaling numerical features where required
- Preparing the data for machine learning models

---

## 📅 Date Feature Engineering

The original date information was converted into useful numerical features.

The following features were extracted:

- `year`
- `month`
- `day`
- `week_of_year`
- `day_of_week`
- `is_weekend`

This allows the models to learn possible temporal patterns in food waste.

In the Streamlit application, the user selects a date and the application automatically derives these date-related features.

---

# 🤖 Machine Learning Models

Three regression models were trained and evaluated:

### 1. Linear Regression

Linear Regression was used as a baseline regression model to understand the relationship between the input variables and food waste.

### 2. Random Forest Regression

Random Forest Regression was used to capture nonlinear relationships between operational conditions and food waste.

### 3. Gradient Boosting Regression

Gradient Boosting Regression was used as another tree-based ensemble model capable of learning complex patterns in the dataset.

---

# 📈 Initial Model Evaluation

Initially, the models were evaluated using a random train-test split.

The initial R² Score results were:

| Model | R² Score |
|---|---:|
| Linear Regression | 0.906 |
| Random Forest | 0.905 |
| Gradient Boosting | 0.917 |

Gradient Boosting achieved the highest R² Score during the initial random-split evaluation.

However, this project focuses on predicting future food waste. A random split can mix observations from different points in time between the training and testing sets.

Therefore, the evaluation methodology was improved using a time-based split.

---

# ⏳ Improved Evaluation Using Time-Based Split

To make the evaluation more realistic, the dataset was sorted chronologically and divided into training and testing data based on time.

Older observations were used for training, while newer observations were used for testing.

**Older Historical Data → Training Data**

**Newer Data → Testing Data**

This approach better represents a real-world scenario where a model is trained using historical information and then used to predict future observations.

---

# 🔬 Model Performance Metrics

The models were evaluated using the following regression metrics:

### MAE — Mean Absolute Error

MAE represents the average absolute difference between the actual and predicted food-waste values.

Lower MAE indicates better performance.

### MSE — Mean Squared Error

MSE calculates the average squared difference between actual and predicted values.

Larger errors receive greater penalty.

Lower MSE indicates better performance.

### RMSE — Root Mean Squared Error

RMSE is the square root of MSE and is expressed in the same unit as the target variable.

Lower RMSE indicates better performance.

### R² Score

R² Score indicates how well the model explains the variation in the target variable.

A value closer to 1 generally indicates stronger explanatory performance.

---

# 📊 Final Time-Based Model Comparison

After using the time-based train-test split, the three models produced the following results:

| Model | MAE (kg) | RMSE (kg) | R² Score |
|---|---:|---:|---:|
| 🏆 **Linear Regression** | **5.222** | **7.863** | **0.919** |
| Random Forest | 5.609 | 8.295 | 0.910 |
| Gradient Boosting | 5.471 | 9.165 | 0.890 |

Based on the time-based evaluation, **Linear Regression achieved the best overall performance**.

Its final test performance was:

- **MAE:** 5.222 kg
- **RMSE:** 7.863 kg
- **R² Score:** 0.919

Therefore, Linear Regression was selected as the final model.

---

# 🔧 Hyperparameter Tuning

Gradient Boosting was further tested using hyperparameter tuning with `RandomizedSearchCV`.

The tuning process explored parameters such as:

- Number of estimators
- Learning rate
- Maximum depth
- Minimum samples split
- Minimum samples leaf

The tuned Gradient Boosting model achieved:

**R² Score: 0.911**

on the time-based test data.

Since this was lower than the Linear Regression R² Score of **0.919**, the tuned Gradient Boosting model was not selected as the final model.

This demonstrates that a more complex model does not necessarily provide better generalization on unseen data.

---

# 🏆 Final Model Selection

## Linear Regression

Linear Regression was selected as the final model because it achieved the best performance across the main evaluation metrics on the time-based test set.

### Final Performance

- **MAE:** 5.222 kg
- **RMSE:** 7.863 kg
- **R² Score:** 0.919

The training-vs-testing evaluation was also performed to check how well the models generalized to unseen data.

The final deployment model was saved as a machine learning pipeline using Joblib.

---

# 📋 Training vs Testing Performance

The models were also compared using their training and testing R² scores.

| Model | Training R² | Testing R² |
|---|---:|---:|
| Linear Regression | 0.836 | 0.906 |
| Random Forest | 0.988 | 0.905 |
| Gradient Boosting | 0.982 | 0.917 |

This comparison helped identify possible overfitting and understand how well each model generalized from training data to unseen testing data.

Gradient Boosting showed a relatively large training-to-testing difference, while Linear Regression showed a smaller difference.

After the time-based evaluation, Linear Regression achieved the strongest overall test performance.

---

# 🔄 Complete Machine Learning Workflow

**Raw Dataset**

↓

**Data Cleaning**

↓

**Data Type Conversion**

↓

**Date Feature Engineering**

↓

**Feature Selection**

↓

**Categorical Encoding**

↓

**Numerical Feature Scaling**

↓

**Initial Random-Split Evaluation**

↓

**Model Comparison**

↓

**Training vs Testing Analysis**

↓

**Hyperparameter Tuning**

↓

**Time-Based Evaluation**

↓

**Final Model Selection**

↓

**Linear Regression**

↓

**Final Model Training**

↓

**Save Model Pipeline**

↓

**Streamlit Application**

↓

**User Input**

↓

**Food Waste Prediction**

↓

**Waste Risk**

↓

**Recommendation**

---

# 🌐 Streamlit Web Application

The trained model has been integrated into an interactive Streamlit web application.

The application allows users to enter operational and environmental information and receive a food-waste prediction.

### User Inputs

The Streamlit application accepts:

- Meals served
- Kitchen staff
- Temperature
- Humidity
- Date
- Special event
- Previous waste
- Staff experience
- Waste category

The selected date is automatically converted into the required date-related features.

---

# 🔮 Prediction Output

The application provides:

### Predicted Food Waste

The expected food waste is displayed in kilograms.

Example:

**Predicted Food Waste: 23.184 kg**

### Waste Risk

The application classifies the predicted waste into:

- LOW
- MEDIUM
- HIGH

### Waste per Meal

The application also calculates the estimated food waste per meal.

### Recommendation

The system provides a recommendation based on the predicted waste level.

For example:

**Monitor preparation quantities and previous leftovers. Small adjustments may help reduce waste.**

---

# 📊 Model Performance Dashboard

The Streamlit application also contains a model-performance section that displays:

- R² Score
- MAE
- MSE
- RMSE
- Model comparison table
- R² comparison chart
- Error comparison chart

This allows users to understand how the selected model performs compared with the other tested models.

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

---

# 📁 Project Structure

    Food-Waste-Prediction/
    │
    ├── README.md
    ├── app.py
    ├── FoodWastePrediction_ML.ipynb
    ├── final_food_waste_model.pkl
    └── food_waste_cleaned.csv

### File Description

| File | Description |
|---|---|
| `FoodWastePrediction_ML.ipynb` | Complete data preprocessing, model training, evaluation, tuning, and model selection |
| `app.py` | Streamlit frontend application |
| `final_food_waste_model.pkl` | Saved final machine learning pipeline |
| `food_waste_cleaned.csv` | Cleaned dataset used for the project |
| `README.md` | Project documentation |

---

# 🚀 How to Run the Project

## 1. Clone the Repository

    git clone https://github.com/Vikhyat1026/Food-Waste-Prediction.git

## 2. Navigate to the Project Folder

    cd Food-Waste-Prediction

## 3. Install Required Libraries

    pip install pandas numpy scikit-learn joblib streamlit

## 4. Run the Streamlit Application

    streamlit run app.py

The Streamlit application will open in your browser.

---

# 📌 Key Results

The main result of this project is the development of a machine learning system capable of predicting food waste using operational, environmental, historical, and categorical information.

Three regression models were compared:

1. Linear Regression
2. Random Forest Regression
3. Gradient Boosting Regression

The initial random-split evaluation showed:

| Model | R² Score |
|---|---:|
| Linear Regression | 0.906 |
| Random Forest | 0.905 |
| Gradient Boosting | 0.917 |

Because the objective is future prediction, a time-based evaluation was then introduced.

The final time-based evaluation produced:

| Model | MAE (kg) | RMSE (kg) | R² Score |
|---|---:|---:|---:|
| 🏆 **Linear Regression** | **5.222** | **7.863** | **0.919** |
| Random Forest | 5.609 | 8.295 | 0.910 |
| Gradient Boosting | 5.471 | 9.165 | 0.890 |

Linear Regression was therefore selected as the final deployment model.

The final model achieved an R² Score of **0.919**, MAE of **5.222 kg**, and RMSE of **7.863 kg** on the time-based test set.

---

# 💡 Why Linear Regression Was Selected

Although Gradient Boosting initially produced the highest R² Score with the random split, the evaluation strategy was improved to better represent future prediction.

After applying the time-based split:

- Linear Regression achieved the highest R² Score.
- Linear Regression achieved the lowest MAE.
- Linear Regression achieved the lowest RMSE.
- The model showed a relatively stable training-to-testing performance.
- The more complex ensemble models did not outperform Linear Regression on the time-based test data.

Therefore, the final model was selected based on **generalization performance on future-like unseen data**, rather than simply selecting the most complex model.

---

# 🎯 Project Outcome

The final system provides an end-to-end machine learning solution:

**Input → Preprocessing → Prediction → Risk Assessment → Recommendation**

A user can provide kitchen and operational conditions through the Streamlit interface, and the system predicts the expected amount of food waste.

The prediction can help users understand potential waste levels and make better-informed food preparation decisions.

---

# 💡 Future Improvements

The project can be further improved by:

- Adding larger and more diverse real-world datasets
- Including restaurant or kitchen-specific information
- Adding food-demand forecasting
- Adding food preparation quantity recommendations
- Adding real-time data integration
- Adding explainable AI techniques such as SHAP
- Adding historical prediction trends
- Adding downloadable prediction reports
- Deploying the Streamlit application online
- Adding automated model retraining when new data becomes available

---

# ⚠️ Limitations

The model's predictions depend on the quality and distribution of the available dataset.

Actual food waste can also be affected by factors that may not be present in the dataset, such as sudden changes in customer demand, food quality issues, kitchen practices, unexpected events, and operational decisions.

Therefore, the system should be considered a **decision-support tool rather than an exact measurement of future food waste**.

---

# 💡 Conclusion

This project demonstrates how machine learning can be applied to food-waste prediction and decision support.

Multiple regression models were trained and evaluated, including Linear Regression, Random Forest Regression, and Gradient Boosting Regression.

The initial evaluation used a random train-test split. However, because food-waste prediction is related to future observations, the evaluation methodology was improved using a chronological time-based split.

Gradient Boosting initially performed best under the random split, but Linear Regression achieved the strongest performance under the time-based evaluation.

Hyperparameter tuning was also performed for Gradient Boosting. The tuned model achieved an R² Score of 0.911, which was still below the Linear Regression result of 0.919.

Therefore, Linear Regression was selected as the final deployment model.

The final system combines the machine learning model with an interactive Streamlit frontend, allowing users to enter kitchen conditions, receive predicted food waste, view the associated risk level, and obtain recommendations.

---

# 👨‍💻 Project

**Food Waste Prediction — Machine Learning Project**

Built using:

**Python | Pandas | NumPy | Scikit-learn | Joblib | Jupyter Notebook | Streamlit**
