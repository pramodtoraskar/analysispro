import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
 
# create an Excel file object
excel = pd.ExcelFile( '/home/ptoraskar/Desktop/RH-IP/PET_PRI_SPT_S1_M.xls' )
 
# parse the first sheet
df = excel.parse( excel.sheet_names[1] )
 
# rename the columns
df = df.rename( columns=dict( zip( df.columns, ['Date','WTI','Brent'] ) ) )

# cut off the first 18 rows because these rows
# contain NaN values for the Brent prices
df = df[18:]

# index the data set by date
df.index = df['Date']
 
# remove the date column after re-indexing
df = df[['WTI','Brent']]

df[['WTI','Brent']][::].plot()

plt.title('Crude Oil Prices')
plt.xlabel('Year')
plt.ylabel('Price [USD]')
plt.savefig('wti_and_brent_all_data3.png',dpi=200)
