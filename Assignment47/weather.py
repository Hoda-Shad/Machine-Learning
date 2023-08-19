import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Dataset/weatherHistory.csv')
# print(data.head(50))
# data['FormattedDate'] = pd.to_datetime(data['FormattedDate'])
# print(data['Formatted Date'])
# print(data.head(50))
# print(data.loc[data['FormattedDate'][2:25]] )
# data.replace({'FormattedDate' : { '2006-04-01' : '91th'}})
k = data['FormattedDate'].str.split(pat = ' ', n = 1, expand = True)
# print(k)
# print(data['FormattedDate'])
data.insert(loc = 0, column = 'Date', value = k[0])
# print(data)
# # data.loc[data["Date"] == '2006-04-01', "Date"] = '91th'
# k=1
# for i in range (0 , len(data.index)-1, 25 ):
#     data['Date'][i:i+23]= 90+k
#     k += 1
# print(data.iloc[100, 0])
import datetime
data_2 = data.iloc[:, [0]]
print(data_2)
for index,row in data_2.iterrows():
    format = "%Y-%m-%d"
    s = row.astype(str)
    print(s)
    dt = datetime.datetime.strptime(s, format)
    tt = dt.timetuple()
    data['Date'] = tt.tm_yday

# print(data)
data2 = data.groupby(['Date'])['Temperature (C)'].mean().reset_index()
# print(data2)
