# Task 2: House Price Prediction using Linear Regression

## Project Overview

This project is part of the Machine Learning training series by Alfidotech. The objective is to predict house prices using a Linear Regression model based on features such as area, number of bedrooms, bathrooms, and location.

The project includes data preprocessing, exploratory data analysis, model training, prediction, and evaluation using standard regression metrics.

## Problem Statement

Build a regression model that can predict the price of a house given relevant input features using supervised learning (regression).

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Dataset

- **File name:** `house_data.csv`
- **Source:** Provided by Alfidotech
- **Features include:** 
  - Area (in sq ft)
  - Bedrooms
  - Bathrooms
  - Location
  - Price (target variable)

## Workflow

1. **Data Loading and Cleaning**
   - Loaded dataset using Pandas
   - Handled missing values (if any)
   - Encoded categorical variables (e.g., Location)

2. **Exploratory Data Analysis (EDA)**
   - Visualized distributions of numerical features
   - Plotted correlation heatmaps and scatter plots
   - Identified key variables affecting price

3. **Model Building**
   - Applied Linear Regression using Scikit-learn
   - Split the data into training and testing sets
   - Trained the model using the `fit()` function

4. **Model Evaluation**
   - Evaluated model using:
     - R² Score
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)

5. **Prediction**
   - Generated predictions on the test set
   - Visualized actual vs. predicted values

## Results

- R² Score on training set: (e.g., 0.85)
- RMSE on test set: (e.g., 22000.50)

Note: The model performed best with features like area and location. Categorical encoding and feature scaling improved prediction accuracy.


