### importing packages required ###
import site
from turtle import color
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import hydralit_components as hc
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from PIL import Image
import altair as alt
import plotly_express as px
import hydralit_components as hc
from streamlit_option_menu import option_menu
import plotly.figure_factory as ff

from sklearn.preprocessing import LabelEncoder,OneHotEncoder, RobustScaler, StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate,cross_val_score,GridSearchCV,KFold,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import auc, roc_curve
import pickle 


### importing data ### 
df=pd.read_csv("C:/Users/ranab/Desktop/myproject/ped2022.csv")

### Menu Home Page ###
Menu = option_menu(None, ["Home","Dataset","Dashboard", "Findings"],icons=['house',"cloud","bar-chart-line","clipboard-check"],menu_icon="cast", default_index=0, orientation="horizontal", styles={"container": {"padding": "0!important", "background-color": "#B0C4DE"},"icon": {"color": "black", "font-size": "25px"}, "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},"nav-link-selected": {"background-color": "#4F6272"},})
if Menu =="Home": st.title('Healthcare Analytics - OB/PED DPT')
Image= Image.open('child.jpg')
if Menu == "Home": st.image(Image,caption='')
if Menu == "Home": st.write("A new acquisition have saved one of the leading hospitals in Kesrouan region from closing down its operations. The new CEO is trying to understand how the hospital was operating and as a start he asked if we can analyze the data, identify the gaps and provide insights to enhance the operations. As a start, we were given the data related to the OB/PED/Maternity department.")
if Menu == "Home": st.write("In the dataset we have 3,818 observations that fall under Age, Gender, Diagnosis, Discharge, Coverage, Nationality, and LOS (Length of Stay)")
if Menu == "Home": st.write("Diagnosis by protocol falls under the ICD10 regulations. ICD10 stands for International Classification of Diseases, 10th revision which is a diagnostic and procedure coding system that all hospitals worlwide use to classify a disease or illness. For confidentiality and simplicity reasons, in this project, and with the supervision of the Medical Team that handed us the data we were able to sum up the diagnosis into more commercial words that can be better understood. Example: A09 - Diarrhoea and gastroenteritis of presumed infectious origin is now Digestive Diseases.")
if Menu== "Dataset": st.write(df)

### Dashboard ###
if Menu=="Dashboard" : st.header("EDA - Visualizations")
col1, col2, col3, col4 = st.columns(4)
if Menu=="Dashboard":col1.metric("Number of Patients","3,818")
if Menu=="Dashboard":col2.metric("Average LOS","2 days")
if Menu=="Dashboard":col3.metric("Age of Patients","0-40 Y")
if Menu=="Dashboard":col4.metric("Data Timeframe","2018-2020")

df.info()
df['Gender'].value_counts()

### Pie Chart 1 ###
col5, col6, col7  = st.columns(3)
if Menu=="Dashboard": col5.markdown("Gender Distribution")
labels= {"Male":"51", "Female":"49"}
sizes = (51,49)
colors = ['#4F6272','#B0C4DE']
explode = (0,0)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal') 
if Menu=="Dashboard": col5.write(fig1)

df["Nationality"].value_counts()
df["LOS"].value_counts()
df["Diagnosis"].value_counts()
df["Coverage"].value_counts()

### Chart 2  Pie ###
if Menu=="Dashboard": col6.markdown("Coverage Distribution")
labels= {"UN":"65", "INSURANCE":"32", "PRIVATE":"3"}
sizes = (65,32,7)
colors = ['#4F6272', '#B0C4DE', '#E6E6FA']
explode = (0,0,0)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal') 
if Menu=="Dashboard": col6.write(fig1)


### Pie Chart 3 ###

labels= {"Syrian":"65", "Lebanese":"35"}
sizes = (65,35)
colors = ['#4F6272', '#B0C4DE']
explode = (0,0)
fig1, ax1 = plt.subplots()
if Menu== "Dashboard":col7.markdown("Nationality Distribution")
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal') 
if Menu=="Dashboard": col7.write(fig1)

palette1= sns.palplot(sns.cubehelix_palette())

col8, col9, col10 = st.columns(3)

#### Barplot Age ####
age = (df.groupby(['Age'])['Gender'].value_counts(normalize=True).rename('percentage').mul(100).reset_index().sort_values('Age'))
fig1=plt.figure(figsize=(10,8))
sns.barplot(x="Age", y="percentage", hue="Gender", data=age, palette="ch:s=.25,rot=-.25")
plt.show()
if Menu== "Dashboard":col8.markdown("Gender Distribution by Age")
if Menu=="Dashboard": col8.pyplot(fig1)

### Age Nationality ###
age = (df.groupby(['Age'])['Nationality'].value_counts(normalize=True).rename('percentage').mul(100).reset_index().sort_values('Age'))
fig1=plt.figure(figsize=(10,8))
sns.barplot(x="Age", y="percentage", hue="Nationality", data=age, palette="ch:s=.25,rot=-.25")
plt.show()
if Menu== "Dashboard":col9.markdown("Nationality Distribution by Age")
if Menu=="Dashboard": col9.pyplot(fig1)

### Barplot 1 ###
import altair as alt

discharge = (df.groupby(['Discharge'])['Nationality'].value_counts(normalize=True).rename('percentage').mul(100).reset_index().sort_values('Nationality'))
fig1=plt.figure(figsize=(10,8))
sns.barplot(x="Nationality", y="percentage", hue="Discharge", palette="ch:s=.25,rot=-.25", data=discharge)
plt.show()
if Menu== "Dashboard":col10.markdown("Discharge vs Nationality")
if Menu=="Dashboard": col10.pyplot(fig1)


col11, col12, col13 = st.columns(3)
#### barplot 3 ####
df.info()
char1= sns.countplot(x="Nationality", data=df).set(title="Nationality Distribution")
gender = (df.groupby(['Nationality'])['Gender'].value_counts(normalize=True).rename('percentage').mul(100).reset_index().sort_values('Nationality'))
fig1= plt.figure(figsize=(10,8))
pal= {'#4F6272':'F','#B7C3F3':"M"}
sns.barplot(x="Nationality", y="percentage", hue="Gender",data=gender, palette='ch:s=.25,rot=-.25')
plt.show()
if Menu== "Dashboard":col11.markdown("Gender vs Nationality")
if Menu=="Dashboard": col11.pyplot(fig1)


df["Discharge"].value_counts()
#### Barplot 2 ####
palette1= sns.palplot(sns.cubehelix_palette())
char1= sns.countplot(x="Nationality", data=df).set(title="Nationality Distribution")
diagnosis = (df.groupby(['Nationality'])['Diagnosis'].value_counts(normalize=True).rename('percentage').mul(100).reset_index().sort_values('Nationality'))
fig1=plt.figure(figsize=(10,8))
sns.barplot(x="Nationality", y="percentage", hue="Diagnosis", data=diagnosis, palette='ch:s=.25,rot=-.25')
plt.show()
if Menu== "Dashboard":col12.markdown("Diagnosis vs Nationality")
if Menu=="Dashboard": col12.pyplot(fig1)


palette1= sns.palplot(sns.cubehelix_palette())
char1= sns.countplot(x="Age", data=df).set(title="Coverage")
age = (df.groupby(['Age'])['Coverage'].value_counts(normalize=True).rename('percentage').mul(100).reset_index().sort_values('Age'))
fig1=plt.figure(figsize=(10,8))
sns.barplot(x="Age", y="percentage", hue="Coverage", data=age, palette='ch:s=.25,rot=-.25')
plt.show()
if Menu== "Dashboard":col13.markdown("Coverage vs Age")
if Menu=="Dashboard": col13.pyplot(fig1)


if Menu == "Findings": st.write("1. Patients privately covered tend to avoid being admitted to the hospital or stay for short periods and leave while they are still sick because of the high cost of treamtment.")
if Menu == "Findings": st.write("2. Although the UN covers the Syrians fully, and they undergo full treatment, yet they tend to get sick more often and babies born are more prone to have parasites from their mothers.")
if Menu == "Findings": st.write("3. 2,035 Syrian patients are under obstetric (delivery) vs 295 Lebanese patients")
if Menu == "Findings": st.write("4. If a girl admitted to the hospital, aged between 15-18 years old, and holds a Syrian nationality, we predict her diagnosis to be Obstetric/Gynecology")
if Menu == "Findings": st.write("The OB/PED Department is admitting a lot of patients that are covered by the UN which means payments are settled in Fresh USD, but although there might be a financial profit, yet this is causing a big problem with staff readiness and safety, overpopulation, and higher mortality rates due to lack of hygiene awareness.")

############################ PREDICTIVE MODELLING #########################################

pd.crosstab(df["Diagnosis"], df["Coverage"])

#### PREDICTIVE MODEL ###
X = df.drop(['Nationality','Coverage', 'Age', 'Gender', "Discharge", "LOS"],axis = 1)
y = df['Discharge']

num_features = X.select_dtypes(include = np.number).columns.tolist()
cat_features = X.select_dtypes(include = 'object').columns.tolist()

lbl = LabelEncoder()
y = lbl.fit_transform(y)

# Categorical Pipeline by replacing missing values with most frequent value and one hot encoding
categorical_pipeline = Pipeline(steps=[('impute',SimpleImputer(missing_values=np.nan, strategy='most_frequent')),('ohe',OneHotEncoder(sparse=False,handle_unknown='ignore'))])
numerical_pipeline = Pipeline(steps=[('imputer', KNNImputer(missing_values=np.NaN)),('normalize',MinMaxScaler())])
preprocessor = ColumnTransformer(transformers =[('num', numerical_pipeline, num_features),('cat', categorical_pipeline, cat_features)], remainder = 'drop')
cv = KFold(n_splits=10, random_state=42, shuffle=True)
model = LogisticRegression()
pipe_lg = Pipeline(steps=[('preprocessor',preprocessor),('model',model)])
scores = cross_validate(pipe_lg, X, y, scoring= ['roc_auc','accuracy'], cv = cv, n_jobs=-1, return_train_score = True)

# Reporting the Results
print('AUC-Train : Cross Validation AUC: %.3f, Standard Deviation: (%.3f)' % (np.mean(scores['train_roc_auc']), np.std(scores['train_roc_auc'])))
print('AUC-Test : Cross Validation AUC %.3f, Standard Deviation: (%.3f)' % (np.mean(scores['test_roc_auc']), np.std(scores['test_roc_auc'])))
print('Accuracy Train : Cross Validation Accuracy: %.3f '% np.mean(scores['train_accuracy']))
print('Accuracy Test : Cross Validation Accuracy: %.3f '% np.mean(scores['test_accuracy']))



