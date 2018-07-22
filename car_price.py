import pandas as pd
import numpy as np
from sklearn import preprocessing,cross_validation
from sklearn.linear_model import LinearRegression

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sb
style.use('ggplot')


df=pd.DataFrame.from_csv("/Users/rashika17/Downloads/data.csv")
df.reset_index(inplace=True)
print(df.head())
df.describe(include='all')
print(df.dtypes)





df.fillna(-99999,inplace=True)
#Converting the numbered categorical data into object type
df['Year']=df['Year'].astype('category')
df['Engine Fuel Type']=df['Engine Fuel Type'].astype('category')
df['Number of Doors']=df['Number of Doors'].astype('category')
df['Engine Cylinders']=df['Engine Cylinders'].astype('category')
df['Market Category']=df['Market Category'].fillna('missing')



df.dropna(inplace=True)
print(df.describe(include='all'))

#vizualization of categorical data using seaborn
plt.plot(figsize=(13,3))
plt.xticks(rotation=90)
sb.stripplot (x='Market Category', y='MSRP',hue='Year', data=df,jitter=True, linewidth=0.8,edgecolor='black',size=8)
plt.show()

plt.plot(figsize=(13,3))
plt.xticks(rotation=90)
sb.stripplot (x='Make', y='Popularity',hue='Year', data=df,jitter=True, linewidth=0.8,edgecolor='black',size=8)
plt.show()

df=pd.get_dummies(df,dummy_na=False,columns=['Make','Model','Engine Fuel Type', 'Transmission Type','Market Category','Driven_Wheels','Vehicle Size','Vehicle Style'])
X=np.array(df.drop('MSRP',axis=1))


y=np.array(df['MSRP'])

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2,random_state=42)

clf=LinearRegression()
clf.fit(X_train,y_train)

print("training done.")
accuracy=clf.score(X_test,y_test)

print(accuracy)






















