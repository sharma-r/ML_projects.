import pandas as pd
import quandl, math
import numpy as np
import datetime
#ML
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

#visualization
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


#Loading the DATA
quandl.ApiConfig.api_key='z_fzBohYvtJ2y6AMpZz-'
df = quandl.get_table("WIKI/PRICES")
#print(df.columns)

#not allfeatures required, remove the redundant ones
df=df[['date','adj_open', 'adj_high', 'adj_low',
       'adj_close', 'adj_volume']]
df.set_index('date',inplace=True)

#defining new features
df['HL_PCT']=(df['adj_high']-df['adj_low'])/df['adj_low']*100
df['PCT_CHNG']=(df['adj_close']-df['adj_open'])/df['adj_open']*100

df=df[['adj_close','HL_PCT','PCT_CHNG','adj_volume']]
#print(df.head(2))



#creating labels
#choose a forecart column
forecast_col='adj_close'
df.fillna(-99999,inplace=True)

#choose no.forecast days.
forecast_out=int(math.ceil(0.01*len(df)))
print('length=',len(df),"and forecast_out=",forecast_out)

#label: shifts adj.close up by forecast_out
df['label']=df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

#print(df.head(2))

# creating feature set
X = np.array(df.drop(['label'], 1))
X=preprocessing.scale(X)

X_lately=X[-forecast_out:]
X=X[:-forecast_out]

y=np.array(df['label'])
y=y[:-forecast_out]

#training and testing,
#dataset splited into training and testing set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

print('length of X_train and x_test: ', len(X_train), len(X_test))

clf=LinearRegression()
clf.fit(X_train,y_train)

accuracy=clf.score(X_test,y_test)
print(accuracy)

#pedictions
forecast_set=clf.predict(X_lately)
print(forecast_set)

df['Forecast']=np.nan
last_date=df.iloc[-1].name
last_unix=last_date.timestamp()
one_day=86400
next_unix=last_unix+one_day

for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]


df['adj_close'].plot(color="green")
df['Forecast'].plot(color="orange")
plt.legend(loc=4)
plt.ylabel('prices')
plt.xlabel('date')
plt.show()


# Zoomed In to a year
df['adj_close'].plot(figsize=(15,6), color="green")
df['Forecast'].plot(figsize=(15,6), color="orange")
plt.xlim(xmin=datetime.date(1986, 4, 26))
plt.ylim(ymin=500)
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()






























