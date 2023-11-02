import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

svc_model_lin = SVC(kernel='linear')
svc_model_lin.fit(X_train, y_train)

# Creating a Logistic Regression model.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Creating a Random Forest Classifier model.
ran_for_class = RandomForestClassifier(n_jobs=-1, n_estimators=100)
ran_for_class.fit(X_train, y_train)

@st.cache()
def prediction(model, sepal_length, sepal_width, petal_length, petal_width):
  pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
  species = pred[0]
  if species == 0: 
    return 'Iris-setosa'
  elif species == 1: 
    return 'Iris-virginica'
  else: 
    return 'Iris-versicolor'

st.sidebar.title("Iris Flower Species Prediction App")

# Add 4 sliders and store the value returned by them in 4 separate variables.
s_len = st.sidebar.slider("Sepal Length", float(iris_df["SepalLengthCm"].min()), float(iris_df["SepalLengthCm"].max()))
s_wid = st.sidebar.slider('Sepal Width', float(iris_df['SepalWidthCm'].min()), float(iris_df['SepalWidthCm'].max()))
p_len = st.sidebar.slider('Petal Length', float(iris_df['PetalLengthCm'].min()), float(iris_df['PetalLengthCm'].max()))
p_wid = st.sidebar.slider('Petal Width', float(iris_df['PetalWidthCm'].min()), float(iris_df['PetalWidthCm'].max()))

classifier = st.sidebar.selectbox('Classifier',('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))

if st.sidebar.button('Predict'): 
  if classifier == 'Support Vector Machine': 
    species_type = prediction(svc_model_lin, s_len, s_wid, p_len, p_wid)
    score = svc_model_lin.score(X_train, y_train)
  elif classifier == 'Logistic Regression':
    species_type = prediction(log_reg, s_len, s_wid, p_len, p_wid)
    score = log_reg.score(X_train, y_train)
  else: 
    species_type = prediction(ran_for_class, s_len, s_wid, p_len, p_wid)
    score = ran_for_class.score(X_train, y_train)
  st.write('Species predicted:', species_type)
  st.write('Accuracy of this model is:', score)