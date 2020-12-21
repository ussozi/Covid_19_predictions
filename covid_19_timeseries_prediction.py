#*********************************USING FACEBOOKS PROPHET PACKAGE******************************************

#importing packages
import pandas as pd
from fbprophet import Prophet

#load data
raw_data=pd.read_csv(r'C:\Users\ssoziu\Desktop\Data Insights\Covid 19 predictions\train.csv')

#Drop unnecesaru columns
raw_data= raw_data.drop(['Territory X Date', 'cases'],axis=1)

#Sort data
raw_data= raw_data.sort_values(['Territory','Date'])

#initialising final dataframe
submission=pd.DataFrame(columns=['Territory','Date', 'target', 'target_lower', 'target_upper'])

#Getting the various country names
countrys=sorted(raw_data['Territory'].unique())

#We run the prediction per country since they have varying trends on deaths
for country in countrys:

    #picking off specific country data
    data=raw_data[raw_data['Territory']==country]

    data=data[['Date','target']]

    #Changing date to datetime index
    data['Date'] = pd.DatetimeIndex(data['Date'])

    #force the column names to be ds and y as per prohet requirements
    data = data.rename(columns={'Date': 'ds',
                            'target': 'y'})

    # set the uncertainty interval to 95% (the Prophet default is 80%)
    my_model = Prophet(
        interval_width=0.95,
        growth='linear',
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.5,
        n_changepoints=200,
        seasonality_mode='multiplicative')

    #fitting model
    my_model.fit(data)

    #Creating future data frame
    future_dates = my_model.make_future_dataframe(periods=67, freq='D')

    #predict into the future
    forecast = my_model.predict(future_dates)

    #Make final dataframe
    subs = pd.DataFrame({
        'Territory': country,
        'Date':forecast['ds'],
        'target':forecast['yhat'],
        'target_lower':forecast['yhat_lower'],
        'target_upper':forecast['yhat_upper']
    })

    #Append to dataframe
    submission=submission.append(subs)
    print('{} done'.format(country ))

#pick off data greater than the start date and smaller than the end date
mask = (submission['Date'] > '2020-12-13') & (submission['Date'] <= '2020-12-20')

#work around the checks with test data

#making a copy of the submision dataframe
fst=submission.loc[mask].copy()

#Handling the Territory X Date column
fst['Territory X Date']=[c + ' X ' + '{dt.month}/{dt.day}/{dt:%y}'.format(dt =d) for c,d in zip(fst['Territory'],fst['Date'])]

#getting the absolute values
fst['pred_target']=fst['target_upper'].abs()

fst['pred_target'] = [round(value) for value in fst['target_upper']]

final= fst[['Territory X Date','pred_target']]

#Picking test data
test=pd.read_csv(r'C:\Users\ssoziu\Desktop\Data Insights\Covid 19 predictions\test.csv')

#****************************** Scoring*********************************#
from sklearn.metrics import mean_absolute_error

# Calculate MAE
print('MAE: ', mean_absolute_error(test['target'], final['pred_target']))

#sending to csv
final.to_csv(r'C:\Users\ssoziu\Desktop\Data Insights\Covid 19 predictions\validation.csv',index=False)

