#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import cohere

# Load the dataset
df = pd.read_csv(r"C:\\Users\\tejes\\Desktop\\Codethon\\realistic_nano_entrepreneur_loans_v2.csv")
df.head()


# In[23]:


df.info()


# In[24]:


df.drop(['Business_Type','Loan_Status'],axis=1,inplace=True)


# In[25]:


df.head()


# In[26]:


## Data Preprocessing
df['Region_Area_Culture'] = df['Region_Area_Culture'].map({'Rural': 0, 'Semi-Urban': 1, 'Urban': 2})
df['Ration_Card'] = df['Ration_Card'].map({'BPL': 0, 'APL': 1})
df['Disability_Health'] = df['Disability_Health'].map({'Healthy': 0, 'Minor Issues': 1, 'Major Issues': 2})
df['Education'] = df['Education'].map({'Primary': 0, 'Secondary': 1, 'Graduate': 2, 'Post-Graduate': 3})
df['Ease_of_Technology'] = df['Ease_of_Technology'].map({'Low': 0, 'Moderate': 1, 'High': 2})
df['Criminal_Offenses'] = df['Criminal_Offenses'].map({'No Offences': 0, 'Single Offence': 1, 'Multiple Offence': 2})
df['Informal_Loans'] = df['Informal_Loans'].map({'No Loans': 0, 'Single Loan': 1, 'Two Loans': 2, 'Three Loans': 3})


# In[27]:


df.head()


# In[31]:


plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),cbar=True,annot=True)


# In[32]:


x=df.drop('Creditworthiness',axis=True) 
y=df['Creditworthiness'] #output variable


# In[34]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)

# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[43]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

# Train the model
regressor = RandomForestRegressor()
regressor.fit(x_train_scaled, y_train)

# Predict on the test set
y_pred = regressor.predict(x_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"Mean Squared Error:",{mse})
print(f"Mean Absolute Error",{mae})
print("r2_score is",r2)


# In[48]:


#Reducing The Overfitting Condition
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Initialize the Lasso model
lasso = Lasso(alpha=0.06)

# Train the model
lasso.fit(x_train_scaled, y_train)

# Predict on the test set
y_pred_lasso = lasso.predict(x_test_scaled)

# Evaluate the model
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"Lasso Mean Squared Error: {mse_lasso}")
print(f"Lasso Mean Absolute Error: {mae_lasso}")
print(f"Lasso R2 Score: {r2_lasso}")


# In[49]:


# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Random Forest Predictions')
plt.scatter(y_test, y_pred_lasso, color='red', label='Lasso Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual Creditworthiness')
plt.ylabel('Predicted Creditworthiness')
plt.title('Actual vs Predicted Creditworthiness')
plt.legend()
plt.show()


# In[73]:


# Initialize the Cohere client
co = cohere.Client("EE8HXeNBQpFVeOpFrVU69NUMXGY0xfYzdY5pEqni")

def generate_application_response(features, reason):
    prompt = f"""
   Output Generation:
                1.Creditworthiness percentage.
                2.Approval status with a congratulatory message or reasons for denial(does not include lack of credit score or lack of document because the nano-entrepreneur doesn't have these as usual but what the bank is doing is if any nano-entrepreneur came, it took data from customer and do analysis with their data with the condition of nano-entrepreneur available in their area by doing a survey and then compare both the data by using ml algorithm whether the nano-entrepreneur is eligible not.
                3.Once again I prefer you, when the application is rejected do not include the reason that the applicant doesn't have the credit score or fewer documents because the nano-entrepreneur does not have these as usual, instead focus on the data provided by the customer and analyze the data with the available data.
                  Applicant details: {features}
                  Accepted or rejected reason: {reason}.
    """
 
    

	
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=800,
        temperature=0.7
    )
    return response.generations[0].text.strip()

def calculate_creditworthiness(applicant_features):
    # Define weights for each parameter
    weights = {
        "Income": 0.2,
        "Expenditure": -0.2,
        "Savings": 0.15,
        "BusinessInvestment": 0.1,
        "Region_Area_Culture": 0.05,
        "Family_Size": 0.05,
        "Ration_Card": 0.05,
        "Disability_Health": 0.1,
        "Government_Aid": 0.05,
        "Education": 0.1,
        "Ease_of_Technology": 0.05,
        "Criminal_Offenses": -0.1,
        "Informal_Loans": -0.1
    }

    # Calculate the score
    score = sum(applicant_features[param] * weight for param, weight in weights.items())
    return score

def process_application(applicant_features):
    # Convert features to DataFrame
    applicant_df = pd.DataFrame([applicant_features])

    # Preprocess the features
    applicant_df['Region_Area_Culture'] = applicant_df['Region_Area_Culture'].map({'Rural': 0, 'Semi-Urban': 1, 'Urban': 2})
    applicant_df['Ration_Card'] = applicant_df['Ration_Card'].map({'BPL': 0, 'APL': 1})
    applicant_df['Disability_Health'] = applicant_df['Disability_Health'].map({'Healthy': 0, 'Minor Issues': 1, 'Major Issues': 2})
    applicant_df['Education'] = applicant_df['Education'].map({'Primary': 0, 'Secondary': 1, 'Graduate': 2, 'Post-Graduate': 3})
    applicant_df['Ease_of_Technology'] = applicant_df['Ease_of_Technology'].map({'Low': 0, 'Moderate': 1, 'High': 2})
    applicant_df['Criminal_Offenses'] = applicant_df['Criminal_Offenses'].map({'No Offences': 0, 'Single Offence': 1, 'Multiple Offence': 2})
    applicant_df['Informal_Loans'] = applicant_df['Informal_Loans'].map({'No Loans': 0, 'Single Loan': 1, 'Two Loans': 2, 'Three Loans': 3})

    # Scale the features
    applicant_scaled = scaler.transform(applicant_df)

    # Calculate creditworthiness
    creditworthiness_score = regressor.predict(applicant_scaled)[0]

    # Define a threshold for loan approval
    threshold = 0.5

    if creditworthiness_score >= threshold:
        response = "Congratulations! Your loan application has been approved. We are excited to support your entrepreneurial journey. Best of luck with your business!"
    else:
        rejection_reason = "Low credit score and insufficient financial documentation."
        response = generate_application_response(applicant_features, rejection_reason)

    return response


# In[82]:


# Test Case 1
applicant_features = {
    "Age": 35,
    "Income": 35000,
    "Expenditure": 20000,
    "Savings": 10000,
    "BusinessInvestment": 5000,
    "Region_Area_Culture": "Urban",
    "Family_Size": 4,
    "Ration_Card": "APL",
    "Disability_Health": "Healthy",
    "Government_Aid": 0,
    "Education": "Graduate",
    "Ease_of_Technology": "Moderate",
    "Criminal_Offenses": "No Offences",
    "Informal_Loans": "No Loans"
}

response = process_application(applicant_features)
print(response)


# In[84]:


# Test Case 2
applicant_features = {
    "Age": 45,
    "Income": 45000,
    "Expenditure": 42500,
    "Savings": 2000,
    "BusinessInvestment": 500,
    "Region_Area_Culture": "Urban",
    "Family_Size": 3,
    "Ration_Card": "APL",
    "Disability_Health": "Healthy",
    "Government_Aid": 0,
    "Education": "Graduate",
    "Ease_of_Technology": "High",
    "Criminal_Offenses": "No Offences",
    "Informal_Loans": "No Loans"
}

response = process_application(applicant_features)
print(response)


# In[81]:


# Test Case 3
applicant_features = {
    "Age": 30,
    "Income": 28000,
    "Expenditure": 220000,
    "Savings": 5000,
    "BusinessInvestment": 1000,
    "Region_Area_Culture": "Semi-Urban",
    "Family_Size": 5,
    "Ration_Card": "BPL",
    "Disability_Health": "Minor Issues",
    "Government_Aid": 1,
    "Education": "Secondary",
    "Ease_of_Technology": "Low",
    "Criminal_Offenses": "Single Offence",
    "Informal_Loans": "Single Loan"
}

response = process_application(applicant_features)

print(response)


# In[ ]:




