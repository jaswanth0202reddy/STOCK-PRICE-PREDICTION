#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df=pd.read_csv('E:\pride/stock price apple.csv')
df


# In[ ]:





# In[4]:


print("first five set of data:")
df.head()


# In[5]:


print("last five set of data:")
df.tail()


# In[6]:


df.info()


# In[10]:


print("data types in dataset:")
df.dtypes


# In[22]:


print("print only data of specific location: ")
print(df.loc[1]['Date'])
print(df.loc[1]['High'])


# In[24]:


print("dataset of first 1000 dataset")
df.iloc[:1001]


# In[2]:


print("no of (rows,columns) in dataset:",df.shape)


# In[45]:


print("showing dataset from highest value of of all time to lowest:")
max_of_high=df.sort_values(by='High', ascending=False)
max_of_high.head(1)


# In[43]:


df2=df["Close"].mean()
print("average closing price of dataset:\n",df2)


# In[50]:


df3=df.describe()
print(df3)


# In[48]:


print("shows the difference of opening and closing of product in stockmarket:")
df4=df['Close']-df['Open']
print(df4)


# In[3]:


print("return the minimun value of each column in dataset:")
df.min()


# In[61]:


print("return the maximum value of each column in dataset:")
df.max()


# In[5]:


import pandas as pd
import numpy as np
df=pd.read_csv('E:\pride/stock price apple.csv')
df.isnull()


# In[7]:


df.notnull()


# In[7]:


df.columns


# In[13]:


df[df['Open']==max(df['Open'])]


# In[14]:


df[df['Open']==min(df['Open'])]


# In[4]:


import matplotlib.pyplot as plt
print(plt.style.available)


# In[7]:


df


# In[5]:


import matplotlib.pyplot as plt


# In[7]:


plt.plot(opening_price,color='red',linewidth=2)
plt.xlabel('Date')
plt.ylabel('Open')
plt.title('apple stock price')
plt.show()


# In[14]:


#line plot
dates=df['Date']
Closing_price=df['Close']
plt.plot(dates,Closing_price,marker='*')
plt.show()


# In[22]:


df.iloc[999]


# In[15]:


df_groupby_year=df.groupby('year')
year=df_groupby_year
avg_close=df_groupby_year['Close'].mean()
avg_close


# In[11]:


plt.plot(avg_close,color='red',linewidth=2)
plt.xlabel('year')
plt.ylabel('average_of_closing')
plt.title('apple stock price')
plt.show()


# In[16]:


avg_open=df_groupby_year['Open'].mean()
avg_open


# In[9]:


plt.plot(avg_close,color='red',linewidth=2)
plt.xlabel('year')
plt.ylabel('average_of_opening')
plt.title('apple stock price')
plt.show()


# In[8]:


plt.plot(avg_close,color='red',label="close")
plt.plot(avg_open,color='blue',label="open")
df_groupby_year=df.groupby('year')
year=df_groupby_year
plt.xlabel('year')
plt.ylabel('measure')
plt.title('apple stock price')
plt.legend(loc="center left")
plt.show()


# In[109]:


plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),(df['Low']+df['High'])/1.0)
plt.xticks(range(0,df.shape[0],365),df['year'].loc[::365])
plt.xlabel('year',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.show()


# In[3]:


plt.figure(figsize = (25,17))
plt.plot(range(df.shape[0]),(df['High'])/1.0)
plt.xticks(range(0,df.shape[0],365),df['year'].loc[::365])
plt.xlabel('year',fontsize=18)
plt.ylabel('high',fontsize=18)
plt.show()
max_of_high=df.sort_values(by='High', ascending=False)
max_of_high.head(1)


# In[42]:


df.max()


# In[25]:


#histogram
avg_high=df_groupby_year['High'].mean()
avg_high


# In[26]:


avg_low=df_groupby_year['Low'].mean()
avg_low


# In[4]:


import seaborn as sns
plt.figure(figsize=(5,5))
sns.boxplot([df['Open'],df['Close']])
sns.boxplot()
p = plt.title('distribution for open and close')
p = plt.ylabel('price')
plt.xticks([0,1.4],['Open','Close'])
plt.show()


# In[18]:


import seaborn as sns
sns.stripplot(x ='Date',y='High',data=df)
plt.show()


# In[32]:


plt.figure(figsize=(20,10))
sns.barplot(data=df ,x='year',y='High')
plt.show()


# In[34]:


plt.figure(figsize=(20,10))
sns.barplot(data=df ,x='year',y='Close')
plt.show()


# In[7]:


plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('days')
plt.ylabel('Close Prices')
plt.plot(df['Close'])
plt.title('ARCH CAPITAL GROUP closing price')
plt.show()


# In[16]:


##Converting Date to DateTime Object
df['Date']


# In[17]:


df['Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')


# In[18]:


##Making Date as Index 
df.set_index('Date',inplace=True)


# In[19]:


df['Date'] = df.index


# In[20]:


df.head()


# In[21]:


col_names = df.columns

fig = plt.figure(figsize=(24, 24))
for i in range(6):
  ax = fig.add_subplot(6,1,i+1)
  ax.plot(df.iloc[:,i],label=col_names[i])
  df.iloc[:,i].rolling(100).mean().plot(label='Rolling Mean')
  ax.set_title(col_names[i],fontsize=18)
  ax.set_xlabel('Date')
  ax.set_ylabel('Price')
  ax.patch.set_edgecolor('black')  
  plt.style.context('fivethirtyeight')
  plt.legend(prop={'size': 12})
  plt.style.use('fivethirtyeight')

fig.tight_layout(pad=3.0)

plt.show()
     


# In[25]:


##HeatMap to Verify Multicollinearity between Features
import seaborn as sns
fig = plt.figure(figsize=(16,12))
matrix = np.triu(df.corr())
ax = sns.heatmap(df.corr(),annot=True,annot_kws={"size":14},mask=matrix,cmap='coolwarm')
ax.tick_params(labelsize=14)
sns.set(font_scale=3)
ax.set_title('HeatMap')
plt.style.use('fivethirtyeight')
plt.show()


# In[26]:


##Data after feature selection
df_feature_selected = df.drop(axis=1,labels=['Open','High','Low','Close','Volume'])


# In[28]:


col_order = ['Date','Adj Close']
df_feature_selected = df_feature_selected.reindex(columns=col_order)
df_feature_selected


# # Resampling

# In[29]:


##Resample Data to Monthly instead of Daily by Aggregating Using Mean
monthly_mean = df_feature_selected['Adj Close'].resample('M').mean()


# In[30]:


monthly_data = monthly_mean.to_frame()
monthly_data


# In[31]:


##Monthly Stock Price 
fig = plt.figure(figsize=(18,8))
plt.plot(monthly_data['Adj Close'],label='Monthly Averages Apple Stock')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.set_title('Monthly Resampled Data')
plt.style.use('fivethirtyeight')
plt.legend(prop={'size': 12})
plt.show()
     


# # Exploratory Data Analysis

# In[33]:


monthly_data['Year'] = monthly_data.index.year
monthly_data['Month'] = monthly_data.index.strftime('%B')
monthly_data['dayofweek'] = monthly_data.index.strftime('%A')
monthly_data['quarter'] = monthly_data.index.quarter
monthly_data


# In[35]:


fig = plt.figure(figsize=(8,6))
sns.boxplot(monthly_data['Adj Close']).set_title('Box Plot Apple Stock Price')
plt.style.context('fivethirtyeight')


# Box-Plot Inference :-
# 
# ->Distribution shows Right Skew
# 
# ->Outlier towards the higher end around Stock price of 300$

# In[40]:


print('Skewness of Distribution is ',monthly_data['Adj Close'].skew())
print('Kurtosis of Distribution is ',monthly_data['Adj Close'].kurtosis())


# In[43]:


plt.figure(figsize=(20,15))
ax = sns.boxplot(x=monthly_data['Year'],y=monthly_data['Adj Close'],palette='RdBu')
ax.set_title('Box Plots Year Wise-Apple Stock Price')
plt.style.context('fivethirtyeight')

Inferences Box Plot

->Outliers Present in Year 2012 and 2019

->Lot of Variability in Years 2014, 2017-19

->2019 most volatile year among all years

->Upward Rising Trend is shown
# In[62]:


group_by_yr = []
list_years = monthly_data['Year'].unique()
dict_IQR = {}
for yr in list_years:
  group_by_yr.append('df' + str(yr)) 

for enum,yr in enumerate(list_years):
   group_by_yr[enum] = monthly_data[str(yr)]['Adj Close']
   dict_IQR[str(yr)] = stats.iqr(group_by_yr[enum])
     


# In[46]:


fig, ax = plt.subplots(figsize=(30,15))
palette = sns.color_palette("mako_r", 4)
a = sns.barplot(x="Year", y="Adj Close",hue = 'Month',data=monthly_data)
a.set_title("Stock Prices Year & Month Wise",fontsize=15)
plt.legend(loc='upper left')
plt.show()

Above figure shows that the Period from July-September seems to push stock price above in comparision to other months. The primary reason for this is as Apple has a product cycle release date during this time,the Wallstreet is excited about upcoming products .
# In[48]:


fig = plt.figure(figsize=(12,16))
fig.set_size_inches(10,16)
group_cols = monthly_data.columns

for enum,i in enumerate(group_cols[1:]):
  ax = fig.add_subplot(4,1,enum+1)
  Aggregated = pd.DataFrame(monthly_data.groupby(str(i))["Adj Close"].mean()).reset_index().sort_values('Adj Close')
  sns.barplot(data=Aggregated,x=str(i),y="Adj Close",ax=ax)
  ax.set(xlabel=str(i), ylabel='Mean Adj Close')
  ax.set_title("Average Stock Price By {}".format(str(i)),fontsize=15)
  plt.xticks(rotation=45)
  
plt.tight_layout(pad=1)

->According to Mean price by Years, 2013 and 2016 are the only years where Mean price is lower than previous Year.

->Average Stock Price is lower at start of the week in comparision to the end of the week.

->The Average Price is Highest in the Month of November.

->Q4 is the best for Apple according to average stock price. By sales figures Q4 has always been strong for Apple since the new product cycle takes place and its the Holiday period. We also observe this as a seasonal effect for Apple.
# # Transformations To Make Series Stationary

# In[49]:


##Differencing By 1
monthly_diff = monthly_data['Adj Close'] - monthly_data['Adj Close'].shift(1)
     


# In[54]:


monthly_diff[1:].plot(c='grey')
monthly_diff[1:].rolling(20).mean().plot(label='Rolling Mean',c='orange')
monthly_diff[1:].rolling(20).std().plot(label='Rolling STD',c='yellow')
plt.legend(prop={'size': 8})


# # Modelling Seasonal ARIMA

# In[55]:


modelling_series = monthly_data['Adj Close']
modelling_series
     


# # Train-Test Split

# In[66]:


train,test = split(modelling_series,train_size=0.6,shuffle=False)


# In[67]:


train.head(2)


# In[73]:


test.head(2)


# In[74]:


print('Train',len(train))
print('Test',len(test))


# In[75]:


p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of grid search Model parameter combinations for Seasonal-ARIMA')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))


# In[76]:


p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of grid search Model parameter combinations for Seasonal-ARIMA')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))


# # Back-Testing Training and Testing Data

# In[91]:


## Using TimeSeriesSplit from sklearn library
time_series_splits = TimeSeriesSplit(n_splits=4)
X = modelling_series.values
plt.figure(1)
fig = plt.figure(figsize=(12, 12))

index = 1
for train_index, test_index in time_series_splits.split(X):
	train = X[train_index]
	test = X[test_index]
	print('Observations: %d' % (len(train) + len(test)))
	print('Training Observations: %d' % (len(train)))
	print('Testing Observations: %d' % (len(test)))


# In[81]:


train_list = {}
test_list = {}
time_series_splits = TimeSeriesSplit(n_splits=5)
X = modelling_series.values

index = 1
for train_index, test_index in time_series_splits.split(X):
    train = X[train_index]
    test = X[test_index]
    train_list[index] = train
    test_list[index] = test
    index += 1
     


# In[82]:


def backtest_model(train,test):
    model = sm.tsa.SARIMAX(train,order=(1,1,1),seasonal_order=(2,2,0,12))
    results=model.fit()


    # train_get_dates_beginning = '2012-01-31'
    # train_get_dates_ending = str(modelling_series.index[len(train)].date())
    # test_get_dates_beginning = train_get_dates_ending
    # test_get_dates_ending = str(modelling_series.index[len(train)+len(test)].date())


    # forecasts_train = results.predict(start=train_get_dates_beginning,end=train_get_dates_ending)
    # forecasts_test = results.predict(start=test_get_dates_beginning,end=test_get_dates_ending)

    forecasts_train = results.predict(start=0,end=len(train))
    forecasts_test = results.predict(start=len(train),end=len(train)+len(test))


    fig,(ax1,ax2) = plt.subplots(2,figsize=(18,10))

    train = pd.DataFrame(train)
    test = pd.DataFrame(test)

    forecasts_train = pd.DataFrame(forecasts_train)
    forecasts_test = pd.DataFrame(forecasts_test)

    forecasts_train.plot(label='Forecasts',ax=ax1,title='SARIMA Forecasting -Train Data')
    train.plot(label='Actual',ax=ax1)
    ax1.set_ylabel('Stock Price')
    ax1.set_xlabel('Time')


    forecasts_test.plot(label='Forecasts',ax=ax2,title='SARIMA Forecasting -Test Data')
    test.plot(label='Actual',ax=ax2)
    ax2.set_ylabel('Stock Price')
    ax2.set_xlabel('Time')


    
    

    ax1.legend()
    ax2.legend()
    plt.tight_layout(pad=2)


# # Backtest Set 1

# In[83]:


## Backtest Set-1
backtest_model(train_list[2],test_list[2])


# # Backtest Set-2

# In[84]:


## Backtest Set-3
backtest_model(train_list[4],test_list[4])


# # Backtest Set-3

# In[85]:


## Backtest Set-4
backtest_model(train_list[5],test_list[5])


# In[105]:


train_list[1]


# In[106]:


## Backtest Set-1
backtest_model(train_list[1],test_list[1])


# In[107]:


backtest_model(train_list[2],test_list[2])


# In[108]:


backtest_model(train_list[3],test_list[3])

from observing the forcasting of data it is profitable to buy the apple stock 
here we did backtest for three set and these graphs are observed as result.
