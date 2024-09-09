#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[5]:


data=pd.read_csv("C:\\Users\\vtu10\\Downloads\\Titanic-Dataset.csv")


# In[6]:


data


# In[7]:


data.head()

Data Preparation:

Load the dataset.
Handle missing values.
Encode categorical variables.
Feature selection.
Split the data into training and testing sets.

# In[8]:


# Fill missing values for 'Age' with the median value
data['Age'].fillna(data['Age'].median(), inplace=True)

# Fill missing values for 'Embarked' with the most common value
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column as it has too many missing values
data.drop(columns=['Cabin'], inplace=True)


# In[9]:


# Convert 'Sex' into numerical values
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# One-hot encode 'Embarked'
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)


# In[10]:


# Select relevant features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
X = data[features]
y = data['Survived']


# In[11]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

accuracy, report, confusion


# In[ ]:




