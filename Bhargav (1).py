#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
room_l=[10,11,12,18,17,15]
room_b=[15,17,18,20,22,21]
df=pd.DataFrame({'r_l':room_l,'r_b':room_b})
print(df)


# In[41]:


df['area']=df['r_l']*df["r_b"]
print(df)


# In[42]:


import numpy as np
df['area']=np.where(df['area']>=300,"big",np.where(df['area']<=170,"small","medium"))
df
df1=pd.get_dummies(data=df)
print(df1)


# In[37]:


age=[18,20,23,19,18,22]
city=["A","B","A","B","C","A"]
df2=pd.DataFrame({"age":age,"city":city})
print(df)
df3=pd.get_dummies(data=df)
print(df3)


# In[38]:


df3=pd.get_dummies(data=df,drop_first=True)
print(df3)


# In[46]:


import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
room_l=[10,11,12,18,17,15]
room_b=[15,17,18,20,22,21]
df=pd.DataFrame({'r_l':room_l,'r_b':room_b})
print(df)
sns.regplot(data=df,x="r_l",y="r_b")


# In[47]:


df=pd.read_csv("Book1.csv")
print(df.head())


# In[51]:


sns.regplot(x="cgpa",y="package",data=df)
plt.show()


# In[52]:


x=df.iloc[:,0:1]
y=df.iloc[:,-1]
print(type(x))
print(type(y))


# In[64]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[68]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("Actual",y_test,"Predicted",y_pred)
print(lr.coef_)
print(lr.intercept_)


# In[70]:


from sklearn import metrics
print("MAE:",metrics.mean_absolute_error(y_test,y_pred))
print("MAE:",metrics.mean_squared_error(y_test,y_pred))
print("R2 Score:",metrics.r2_score(y_test,y_pred))


# In[78]:


y_pred=lr.predict([[9.85]])
print(y_pred)


# In[95]:


df=pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")
sns.regplot(x="km_driven",y="selling_price",data=df)
plt.show()


# In[107]:


from sklearn.model_selection import train_test_split
x=df[["km_driven"]]
y=df["selling_price"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)


# In[108]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("Actual",y_test,"Predicted",y_pred)
print(lr.coef_)
print(lr.intercept_)


# In[111]:


y_pred=lr.predict([[35000]])
print(y_pred)


# In[21]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("olympic100.csv")
sns.regplot(x="year",y="time",data=df)


# In[22]:


from sklearn.model_selection import train_test_split
x=df[['year']]
y=df['time']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=5)


# In[23]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("Actual",y_test,"Predicted",y_pred)
print(lr.coef_)
print(lr.intercept_)


# In[24]:


y_pred1=lr.predict([[2024]])
print(y_pred1)


# In[25]:


from sklearn import metrics
print("MSE: ",metrics.mean_squared_error(y_test,y_pred))
print("MAE: ",metrics.mean_absolute_error(y_test,y_pred))
print("R2_score: ",metrics.r2_score(y_test,y_pred))


# In[28]:


df1=pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")
df1.head()
df1["Age"]=2024-df1["year"]


# In[34]:


x=df1[["Age","km_driven","fuel","seller_type","transmission"]]
y=df1["selling_price"]
x=pd.get_dummies(x)
print(x)


# In[31]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)


# In[32]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("Actual",y_test,"Predicted",y_pred)
print(lr.coef_)
print(lr.intercept_)


# In[35]:


from sklearn import metrics
print("MSE: ",metrics.mean_squared_error(y_test,y_pred))
print("MAE: ",metrics.mean_absolute_error(y_test,y_pred))
print("R2_score: ",metrics.r2_score(y_test,y_pred))


# In[48]:


df2=pd.read_csv("winequalityN.csv")
# clean the data set and after that check multipal
df2.isna().sum()


# In[53]:


data=df2.dropna()


# In[54]:


x=data[["type","fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]
y=data["quality"]
x=pd.get_dummies(x)
print(x)


# In[55]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)


# In[56]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("Actual",y_test,"Predicted",y_pred)
print(lr.coef_)
print(lr.intercept_)


# In[57]:


from sklearn import metrics
print("MSE: ",metrics.mean_squared_error(y_test,y_pred))
print("MAE: ",metrics.mean_absolute_error(y_test,y_pred))
print("R2_score: ",metrics.r2_score(y_test,y_pred))


# In[58]:


# ------------------------Polynomial Reg.--------------------------------#


# In[59]:


df=pd.read_csv("data_poly.csv")
print(df)


# In[60]:


x=df.iloc[:,1:2]
y=df.iloc[:,2]


# In[62]:


from sklearn.linear_model import LinearRegression


# In[63]:


from sklearn.preprocessing import PolynomialFeatures


# In[64]:


poly=PolynomialFeatures(degree=3)
model_poly=poly.fit_transform(x)
lr=LinearRegression()
lr.fit(model_poly,y)
plt.scatter(x,y)
plt.plot(x,lr.predict(model_poly))
plt.show()


# In[66]:


lr.predict(poly.fit_transform([[80]]))


# In[67]:


df=pd.read_csv("olympic100.csv")


# In[68]:


from sklearn.model_selection import train_test_split
x=df[['year']]
y=df['time']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=5)


# In[78]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=3)
model_poly=poly.fit_transform(x_train)
lr=LinearRegression()
lr.fit(model_poly,y_train)

plt.scatter(x_train,y_train)
plt.plot(x_train,lr.predict(model_poly))
plt.show()


# In[81]:


lr.predict(poly.fit_transform([[2036]]))


# In[ ]:




