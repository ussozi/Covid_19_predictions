
#import packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy
import statsmodels
import sklearn
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

#Define Functions
def rmse(y_true, y_pred):
   return np.sqrt(mean_squared_error(y_true, y_pred))

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
 
# create a differenced series
def difference(dataset, interval=1):
	diff = []
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)

#load data
data=pd.read_csv(r'C:\Users\ssoziu\Desktop\Data Insights\Covid 19 predictions\train.csv',parse_dates=['Date'],index_col=['Date'])

#Drop unnecesaru columns
data= data.drop(['Territory X Date', 'cases'],axis=1)

#Sort data
data= data.sort_values(['Territory','Date'])

#Checking for seasonality in the data
X = data['target'].values
X = X.astype('float32')

# difference data
days = 1
stationary = difference(X, days)
# check if stationary
result = adfuller(stationary)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#Plotting tthe ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot
pyplot.figure()
pyplot.subplot(211)
plot_acf(stationary, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(stationary, ax=pyplot.gca())
pyplot.show()