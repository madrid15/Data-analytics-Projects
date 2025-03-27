#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries 

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler() 
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.svm import SVC
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import precision_recall_curve,average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import auc
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
import shap
from sklearn.metrics import plot_confusion_matrix


# ## Upload Datasets

# In[3]:


df=pd.read_csv("Thyroid_Diff.csv")


# In[4]:


df


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.shape


# In[9]:


df.describe()


# In[10]:


df.columns


# In[11]:


for column in df.columns:
    unique_values = df[column].unique()
    print(f"Unique values in column '{column}':")
    print(unique_values)
    print()


# ## Age Distribution of Patients

# In[12]:



plt.figure(figsize=(10,10))
sns.histplot(x='Age',hue='Gender',data=df,kde=True,color='blue',edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('count')
plt.show()


# In[13]:


plt.figure(figsize=(30,15))
sns.countplot(x='Age',data=df)
plt.xlabel('Age')
plt.ylabel('Count')


# In[14]:


plt.figure(figsize=(10,10))
sns.histplot(x='Gender',hue='Recurred',data=df,color='darkblue',edgecolor='black')
plt.title('Gender vs Recurred')
plt.xlabel('Age')
plt.ylabel('count')
plt.show()


# In[15]:


colors=['red','darkblue','yellow']
plt.figure(figsize=(10,10))
sns.histplot(x='Age',hue='Risk',data=df,color=colors,edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('count')
plt.show()


# ## Tranform categorical label to numerical label

# In[16]:


le=LabelEncoder()


# In[17]:


for column in df.columns:
    if df[column].dtype==object:
        df[column]=le.fit_transform(df[column])
    


# In[18]:


df


# In[19]:


df.rename(columns={'Hx Smoking': 'Smoking History',
                   'Hx Radiothreapy': 'Radiotherapy History',
                   'Pathology': 'Types of Thyroid Cancer (Pathology)',
                   'T': 'Tumor',
                   'N': 'Lymph Nodes',
                   'M': 'Cancer Metastasis',
                  'Response' : 'Treatment Response'}, inplace=True)


# In[20]:


df


# In[21]:


df1=df.drop("Recurred",axis=1)


# In[22]:


x=df1
y=df['Recurred']


# ## Split data set

# In[23]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=76)


# In[24]:


x_train.shape,x_test.shape


# In[25]:


scaler=StandardScaler()
x_scale_train=scaler.fit_transform(x_train)
x_scale_test=scaler.transform(x_test)


# In[26]:


clf=DecisionTreeClassifier(max_depth=10,min_samples_split=100,criterion='gini')
clf.fit(x_scale_train,y_train)
predict=clf.predict(x_scale_test)
dt_accuracy=(accuracy_score(y_test,predict))
dt_cm=plot_confusion_matrix(clf,x_scale_test,y_test,display_labels=['yes','no'])
dt_report=classification_report(y_test,predict)


# In[27]:


print("dt_accuracy:",dt_accuracy)
print(dt_cm)
print(dt_report)


# In[28]:


y_probs=clf.predict(x_scale_test)


# In[29]:


fpr,tpr,_=roc_curve(y_test,y_probs[:,])
roc_auc=auc(fpr,tpr)
plt.figure(figsize=(8,8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
print('AUC:', roc_auc)


# In[30]:


xgb_model=XGBClassifier()
param_grid={
    'max_depth':[None, 5, 10, 15, 20],
    'n_estimators':[50,100,150,200,250,300],
    'learning_rate':[0.1,0.001,0.0001,0.5],
    'sampling_method':['uniform','gradient_based'],
    'booster':['gbtree', 'gblinear']    
}
grid=GridSearchCV(estimator=xgb_model,param_grid=param_grid,cv=3,n_jobs=-1)
grid.fit(x_scale_train,y_train)


# In[31]:


best_params=grid.best_params_
print("best_params:",best_params)


# In[32]:


best_xgb_estimator=grid.best_estimator_
predict=best_xgb_estimator.predict(x_scale_test)
xgb_accuracy=accuracy_score(y_test,predict)
xgb_cm=plot_confusion_matrix(grid,x_scale_test,y_test,display_labels=['yes','no'])
xgb_report=classification_report(y_test,predict)
print("xgb_accuracy:",xgb_accuracy)
print(xgb_cm)
print(xgb_report)


# In[33]:


proba=best_xgb_estimator.predict(x_scale_test)
fpr,tpr,_=roc_curve(y_test,proba)
roc_auc=auc(fpr,tpr)
print(roc_auc)


# In[40]:


probability=grid.predict(x_scale_test)
fpr,tpr,_=roc_curve(y_test,probability)
roc_auc=auc(fpr,tpr)
plt.figure(figsize=(8,8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
print('AUC:', roc_auc)


# In[35]:


rf=RandomForestClassifier()
rf.fit(x_scale_train,y_train)
y_pred=rf.predict(x_scale_test)
rf_accuracy=accuracy_score(y_pred,y_test)
rf_cm=plot_confusion_matrix(rf,x_scale_test,y_test,display_labels=['yes','no'])
rf_report=classification_report(y_pred,y_test)
print("rf_accuracy:",rf_accuracy)
print(rf_cm)
print(rf_report)


# In[36]:


probability=rf.predict(x_scale_test)
fpr,tpr,_=roc_curve(y_test,probability)
roc_auc=auc(fpr,tpr)
plt.figure(figsize=(8,8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
print('AUC:', roc_auc)


# In[37]:


ada=AdaBoostClassifier(n_estimators=100)
ada.fit(x_scale_train,y_train)
y_pred=ada.predict(x_scale_test)
ada_accuracy=accuracy_score(y_pred,y_test)
print("ada_accuracy:",ada_accuracy)


# In[38]:


ada_cm=plot_confusion_matrix(ada,x_scale_test,y_test,display_labels=['yes','no'])
ada_report=classification_report(y_pred,y_test)
print(ada_cm)
print(ada_report)


# In[39]:


y_proba=ada.predict(x_scale_test)
fpr,tpr,_=roc_curve(y_test,y_proba)
roc_auc=auc(fpr,tpr)
print(roc_auc)
plt.figure(figsize=(8,8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
print('AUC:', roc_auc)


# In[42]:


plt.figure(figsize=(10,8))
labels=['Decision Tree','Random Forest','XG Boost','Ada Boost']
accuracy=[dt_accuracy,rf_accuracy,xgb_accuracy,ada_accuracy]
plt.bar(labels,accuracy,color=['blue','red','orange','yellow'])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuries")
plt.show()


# In[43]:


importances=rf.feature_importances_


# In[45]:


indices = np.argsort(importances)[::-1]
sorted_feature_names = [importances[i] for i in indices]


# In[177]:


plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(x.shape[1]), importances[indices])
plt.xticks(range(x.shape[1]), sorted_feature_names, rotation=45, ha='right')
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()


# In[ ]:


#for decision tree


# In[49]:


precision=(55/55+0)
print(precision)


# In[50]:


recall=55/(55+3)
print(recall)


# In[51]:


f_score = 2 * precision * recall / (precision + recall)
print(f_score)


# In[52]:


# for xg boost


# In[53]:


precision2=(54/(54+1))
precision2


# In[54]:


recall2=54/(54+2)
print(recall2)


# In[55]:


f_score=2 * precision2 * recall2/(precision2+recall2)
f_score


# In[56]:


# for random forest


# In[57]:


precision3=(55/(55+0))
precision3


# In[58]:


recall3=(55/(55+2))
recall3


# In[59]:


f_score= 2 * precision3 * recall3/(precision3+recall3)
f_score


# In[60]:


#for ada boost


# In[61]:


precision4=(51/(51+4))
precision4


# In[62]:


recall4=(51/(51+3))
recall4


# In[63]:


f_score= 2 * precision4 * recall4/(precision4+recall4)
f_score


# In[66]:


import matplotlib.pyplot as plt

# Sample recall scores
recall_scores = [recall, recall2, recall3, recall4]
labels = ['Decision Tree', 'XG Boost', 'Random Forest', 'Ada Boost']

# Create a pie chart
plt.figure(figsize=(15, 8))
plt.pie(recall_scores, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Recall Scores Distribution Among Different Models')
plt.show()


# ## After Exploratory Data Analysis part, we initially went to the classification work. To classify a patient is cure or not we use several types of machine learning models and random forest fit well and boosting an impressive recall score of 0.9649. This model not only classify the patient cure or not but also check which features are mostly impact in thyroid cancer and the model found that treatment response, risk, Age, Size of Tumour, stage of the cancer is mostly important feature. The higher age of the patient less chance of the cure from the cancer. Also, in the Phase of III & Phase IV stage of cancer is less cure rate. Although the patient are responds from the treatment really well.
# ## In medical data analytics field lot of time we found imbalance data set. In imbalance data set we found that there is one class label dominating the other class label. In such cases, model might achieve high accuracy simply by predicting the majority of the class most of the time. However, this would not be helpful in identifying the minority class which often the primary concern. On the other hand recall score means True positive/(True positive + False Negative)
# ## So, recall score basically explain positive class that means how my model correctly said that how many patients had cancer. The most important think is if the false negative value is much lower than the recall score is higher than it can be said that the model classifies the patient really well. So, the basic difference of recall score and accuracy score is accuracy measure the overall correctness considering both positive and negative prediction where recall only focus the positive instance. Secondly, Accuracy can be misleading with imbalance data set whether recall address the performance of the minority class. When accuracy provides a general measure of model’s performances recall is essential for ensuring that the positive cases are not missed. That’s why in my project I used recall score over the accuracy.

# In[ ]:




