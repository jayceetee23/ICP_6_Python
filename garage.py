import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

train = pd.read_csv('train.csv')
print(train)

#Working with Numeric Features
numeric_features = train.select_dtypes(include=[np.number])

corr = numeric_features.corr()              # correlation

print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])

quality_pivot = train.pivot_table(index='GarageArea',
                                  values='SalePrice', aggfunc=np.median)
print(quality_pivot)

#Notice that the median sales price strictly increases as Overall Quality increases.    #quality of life for graph
quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Garage Area')
plt.ylabel('Sale Price')
plt.xticks(rotation=0)
plt.show()

# Scatter plot of GarageArea vs. SalePrice
train = train[train['GarageArea'] < 1200]

GarageArea = train.GarageArea
SalePrice = train.SalePrice


plt.scatter(GarageArea, SalePrice, alpha=.75,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Garage Area')
plt.ylabel('Sale Price')
plt.title('Comparing Garage Area to Sale Price')
plt.show()


