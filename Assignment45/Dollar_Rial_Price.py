import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from LLS import LLs
import re


df = pd.read_csv('Dollar_Rial_Price_Dataset.csv')
# print(df.head(100))

df_ahmadinejad = df.loc[df["Persian_Date"].between("1390/01/01", "1392/05/12")]
df_rohani = df.loc[df["Persian_Date"].between("1390/05/13", "1400/05/12")]
df_raeesi = df.loc[df["Persian_Date"].between("1400/05/13", "1402/05/12")]
# print(df_ahmadinejad.head(10))
# x = ["Ahmadinejad", "Rohani" , "Raeesi"]
# highest_doller_price= []
# highest_doller_price.append(df_ahmadinejad['High'].max())
# highest_doller_price.append(df_rohani['High'].max())
# highest_doller_price.append(df_raeesi['High'].max())
#
# plt.bar(x , highest_doller_price)
# plt.xlabel("Presidential period")
# plt.ylabel("Maximum Dollar Price")
# plt.title("Highest Dollar Price")
# plt.show()
#
# lowest_doller_price= []
# lowest_doller_price.append(df_ahmadinejad['Low'].min())
# lowest_doller_price.append(df_rohani['Low'].min())
# lowest_doller_price.append(df_raeesi['Low'].min())
# print(lowest_doller_price)
# plt.bar(x , lowest_doller_price)
# plt.xlabel("Presidential period")
# plt.ylabel("Minimum Dollar Price")
# plt.title("Lowest Dollar Price")
# plt.show()

df_ahmadinejad.loc [:,"Date"] = df_ahmadinejad.loc [:,"Date"].apply(pd.to_datetime)
df_ahmadinejad.loc [:,"Date"] = pd.to_numeric(df_ahmadinejad.loc [:,"Date"] , errors='coerce')
print(df_ahmadinejad.head(10))