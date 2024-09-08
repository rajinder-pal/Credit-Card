#!/usr/bin/env python
# coding: utf-8

# In[2]:




# In[4]:




# In[18]:


from sklearn.ensemble import RandomForestClassifier


# In[20]:


model = RandomForestClassifier(n_estimators=100, random_state=42)


# In[22]:


import pickle


# In[24]:


with open('fraud_detection_model.pkl', 'wb') as f:
    pickle.dump(model, f)


# In[151]:


from flask import Flask, request, jsonify


# In[153]:


import pickle


# In[155]:


import numpy as np


# In[157]:


app = Flask(__name__)


# In[159]:


model = pickle.load(open('fraud_detection_model.pkl', 'rb'))


@app.route('/')
def home():
    return "Welcome to the Credit Card Fraud Detection App!"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if 'features' not in data:
        return jsonify({'error': 'Missing features'}), 400
    if not all(isinstance(value, float) for value in data['features']):
        return jsonify({'error': 'Invalid data format'}), 400
    input_data = np.array([data['features']])
    prediction = model.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})


# In[163]:


if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:


app.run()


# In[ ]:




# In[2]:




# In[4]:


import pickle


# In[8]:


from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

# In[10]:
import pandas as pd

import pickle


# In[14]:


data = pd.read_csv('../data/creditcard.csv')


# In[20]:


X = data.drop('Class', axis=1)
y = data['Class']


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[24]:


scaler = StandardScaler()


# In[26]:


X_train = scaler.fit_transform(X_train)


# In[28]:


X_test = scaler.transform(X_test)


# In[30]:


model = RandomForestClassifier()


# In[32]:


model.fit(X_train, y_train)


# In[33]:


pickle.dump(model, open('fraud_detection_model.pkl', 'wb'))


# In[36]:




# In[38]:




# In[ ]:




