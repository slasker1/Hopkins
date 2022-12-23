import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from pylab import rcParams

df = pd.read_csv("nyc_real_estate.csv")

ts_nyc = df[['Date','Value']]

ts_nyc = ts_nyc.dropna()

ts_nyc.Date = pd.to_datetime(ts_nyc.Date)

ts1_nyc = ts_nyc.set_index('Date').resample('M').mean()
print (ts1_nyc)

ts1_nyc.plot(figsize = (15, 6))
plt.show()

rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(ts1_nyc, model='additive')
fig = decomposition.plot()
plt.show()

print(ts_nyc)