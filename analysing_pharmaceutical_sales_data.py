
# Commented out IPython magic to ensure Python compatibility.
# Pandas - Data manipulation and analysis library
import pandas as pd
# NumPy - mathematical functions on multi-dimensional arrays and matrices
import numpy as np
# Matplotlib - plotting library to create graphs and charts
import matplotlib.pyplot as plt
# Re - regular expression module for Python
import re
# Calendar - Python functions related to the calendar
import calendar

# Manipulating dates and times for Python
from datetime import datetime

# Scikit-learn algorithms and functions
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor

# Settings for Matplotlib graphs and charts
from pylab import rcParams
rcParams['figure.figsize'] = 12, 8

# Display Matplotlib output inline
# %matplotlib inline

# Additional configuration
np.set_printoptions(precision=2)

"""# On which day of the week is the second drug (M01AE) most often sold?

Loading our sales daily data set from csv file using Pandas.
"""

df = pd.read_csv("data/salesdaily.csv")

"""Let's look at the data."""

print(df.head())

"""Grouping the second drug sales by weekday name."""

df = df[['M01AE', 'Weekday Name']]
result = df.groupby(['Weekday Name'], as_index=False).sum().sort_values('M01AE', ascending=False)

"""Taking the weekday name with most sales and the volume of sales from the result"""

resultDay = result.iloc[0,0]
resultValue = round(result.iloc[0,1], 2)

"""Printing the result"""

print('The second drug, M01AE, was most often sold on ' + str(resultDay))
print('with the volume of ' + str(resultValue))

"""# Which three drugs have the highest sales in Jan 2015, Jul 2016, Sep 2017

Loading monthly sales data set from csv file using Pandas.
"""

df = pd.read_csv("data/salesmonthly.csv")

"""Let's look at the data."""

print(df.head())

"""Because we will be repeating the same calculations for different months and years it is a good idea to write a function"""

def top3byMonth(month, year):
    """
    given a month and a year
    find top 3 drugs sold
    """
    month = str(month) if (month > 9) else '0'+str(month)
    year = str(year)
    # filter by date
    sales = df.loc[df['datum'].str.contains('^'+year+'\-'+month+'', flags=re.I, regex=True)]
    # reset index
    sales = sales.reset_index()
    # filter relevant columns
    topSales = sales[['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']]
    # sort values horizontally
    print(topSales)
    topSales = topSales.sort_values(by=0, ascending=False, axis=1)
    # print results
    print('Top 3 drugs by sale in '+calendar.month_name[int(month)]+' '+year)
    for field in topSales.columns.values[0:3]:
        print(' - Product: ' + str(field) + ', Volume sold: ' + str(round(topSales[field].iloc[0], 2)))
    print("\n")

"""We are now calling the function fir different months and years and printing results"""

# top3 drugs by sale in January 2017
top3byMonth(1, 2017)

# top3 drugs by sale in July 2018
top3byMonth(7, 2018)

# top3 drugs by sale in September 2019
top3byMonth(9, 2019)

"""# Which drug has sold most often on Mondays in 2020?

Loading our sales daily data set from csv file using Pandas.
"""

df = pd.read_csv("data/salesdaily.csv")

"""Let's look at the data."""

df.head()

"""Filtering out from the data everything else apart from year 2018 and Monday"""

df = df.loc[df['datum'].str.contains('2018', flags=re.I, regex=True) & (df['Weekday Name'] == 'Monday')]

"""Groupping by weekday name and summarising"""

df = df.groupby(['Weekday Name'], as_index=False).sum()

"""Filtering only relevant columns and sorting values of most sold drugs horizontally to achieve the most often sold drug on the left"""

df = df[['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']]
result = df.sort_values(by=0, ascending=False, axis=1)

"""Displaying results"""

for field in result.columns.values[0:1]:
    print('The drug most often sold on Mondays in 2018 is ' + str(field))
    print('with the volume of ' + str(round(result[field].iloc[0], 2)))

"""# What medicine sales may be in January 2021?

Defining the scattering function that will display scattered sales data on the chart
"""

def scatterData(X_train, y_train, X_test, y_test, title):
    plt.title('Prediction using ' + title)
    plt.xlabel('Month sequence', fontsize=20)
    plt.ylabel('Sales', fontsize=20)

    # Use Matplotlib Scatter Plot
    plt.scatter(X_train, y_train, color='blue', label='Training observation points')
    plt.scatter(X_test, y_test, color='cyan', label='Testing observation points')

"""Defining predict sales and display Linear Regression model function"""

def predictLinearRegression(X_train, y_train, X_test, y_test):

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    scatterData(X_train, y_train, X_test, y_test, 'Linear Regression')

    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    plt.plot(X_train, reg.predict(X_train), color='red', label='Linear regressor')
    plt.legend()
    plt.show()

    # LINEAR REGRESSION - Predict/Test model
    y_predict_linear = reg.predict(X_test)

    # LINEAR REGRESSION - Predict for January 2021
    linear_predict = reg.predict([[predictFor]])
    # linear_predict = reg.predict([[predictFor]])[0]

    # LINEAR REGRESSION - Accuracy
    accuracy = reg.score(X_train, y_train)

    # LINEAR REGRESSION - Error
    # error = round(np.mean((y_predict_linear-y_test)**2), 2)
    
    # Results
    print('Linear Regression: ' + str(linear_predict) + ' (Accuracy: ' + str(round(accuracy*100)) + '%)')

    return {'regressor':reg, 'values':linear_predict}

"""Defining predict sales and display Polynomial Regression model function"""

def predictPolynomialRegression(X_train, y_train, X_test, y_test):

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    scatterData(X_train, y_train, X_test, y_test, 'Polynomial Regression')
    
    poly_reg = PolynomialFeatures(degree = 2)
    X_poly = poly_reg.fit_transform(X_train)
    poly_reg_model = linear_model.LinearRegression()
    poly_reg_model.fit(X_poly, y_train)
    plt.plot(X_train, poly_reg_model.predict(poly_reg.fit_transform(X_train)), color='green', label='Polynomial regressor')
    plt.legend()
    plt.show()

    # Polynomial Regression - Predict/Test model
    y_predict_polynomial = poly_reg_model.predict(X_poly)

    # Polynomial Regression - Predict for January 2021
    polynomial_predict = poly_reg_model.predict(poly_reg.fit_transform([[predictFor]]))

    # Polynomial Regression - Accuracy
    # X_poly_test = poly_reg.fit_transform(X_test)
    accuracy = poly_reg_model.score(X_poly, y_train)

    # Polynomial Regression - Error
    # error = round(np.mean((y_predict_polynomial-y_train)**2), 2)

    # Result
    print('Polynomial Regression: ' + str(polynomial_predict) + ' (Accuracy: ' + str(round(accuracy*100)) + '%)')
    return {'regressor':poly_reg_model, 'values':polynomial_predict}

"""Defining predict sales and display Simple Vector Regression (SVR) function"""

def predictSVR(X_train, y_train, X_test, y_test):

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    scatterData(X_train, y_train, X_test, y_test, 'Simple Vector Regression (SVR)')

    svr_regressor = SVR(kernel='rbf', gamma='auto')
    svr_regressor.fit(X_train, y_train.ravel())

    # plt.scatter(X_train, y_train, color='red', label='Actual observation points')
    plt.plot(X_train, svr_regressor.predict(X_train), label='SVR regressor')
    plt.legend()
    plt.show()

    # Simple Vector Regression (SVR) - Predict/Test model
    y_predict_svr = svr_regressor.predict(X_test)

    # Simple Vector Regression (SVR) - Predict for January 2021
    svr_predict = svr_regressor.predict([[predictFor]])

    # Simple Vector Regression (SVR) - Accuracy
    accuracy = svr_regressor.score(X_train, y_train)

    # Simple Vector Regression (SVR) - Error
    # error = round(np.mean((y_predict_svr-y_train)**2), 2)
    
    # Result
    print('Simple Vector Regression (SVR): ' + str(svr_predict) + ' (Accuracy: ' + str(round(accuracy*100)) + '%)')
    return {'regressor':svr_regressor, 'values':svr_predict}

"""We are defining a product that we will be predicting the January 2021 sales for.
We can change it to a differnt one and use the same calculations for a different product.
"""

product = 'N02BA'

"""For storing all regression results"""

regResults = pd.DataFrame(columns=('Linear', 'Polynomial', 'SVR', 'Voting Regressor'), index=[product])

"""To display a larger graph than a default with specify some additional parameters for Matplotlib library."""

rcParams['figure.figsize'] = 12, 8

"""We will be using monthly data for our predictions"""

df = pd.read_csv("data/salesmonthly.csv")

"""We will use monthly sales data from 2017, 2018, 2019. We could also use just 2019 for that."""

df = df.loc[df['datum'].str.contains("2015") | df['datum'].str.contains("2016") | df['datum'].str.contains("2017") | df['datum'].str.contains("2018") | df['datum'].str.contains("2019") | df['datum'].str.contains("2020")]
df = df.reset_index()



df['datumNumber'] = 1
for index, row in df.iterrows():
    df.loc[index, 'datumNumber'] = index+1

"""Removing the first and the last incompleted record from Pandas Data Frame"""

# the first and the last available month is quite low which may indicate that it might be incomplete
# and skewing results so we're dropping it
df.drop(df.head(1).index,inplace=True)
df.drop(df.tail(1).index,inplace=True)

"""Cleaning up any rows with the product value = 0."""

df = df[df[product] != 0]

"""Let's look at the data again."""

df.head()

"""What value we predict for? January 2021. Because we have data until August 2020 we're predicting for 5 months ahead"""

predictFor = len(df)+5
print('Predictions for the product ' + str(product) + ' sales in January 2021')

"""For storing regression results."""

regValues = {}

"""Preparing training and testing data by using train_test_split function. 70% for training and 30% for testing."""

dfSplit = df[['datumNumber', product]]

# We are going to keep 30% of the dataset in test dataset
train, test = train_test_split(dfSplit, test_size=3/10, random_state=0)

trainSorted = train.sort_values('datumNumber', ascending=True)
testSorted = test.sort_values('datumNumber', ascending=True)

X_train = trainSorted[['datumNumber']].values
y_train = trainSorted[product].values
X_test = testSorted[['datumNumber']].values
y_test = testSorted[product].values

"""Performing feature scaling. Scaling the feature will improve the performance of the model."""

# scale_X = StandardScaler()
# scale_y = StandardScaler()

# X_train = scale_X.fit_transform(X_train)
# y_train = scale_y.fit_transform(y_train.reshape(-1, 1))

# X_test = scale_X.fit_transform(X_test)
# y_test = scale_y.fit_transform(y_test.reshape(-1, 1))

"""Performing and saving results for Linear Regression"""

# LINEAR REGRESSION
linearResult = predictLinearRegression(X_train, y_train, X_test, y_test)
reg = linearResult['regressor']
regValues['Linear'] = round(linearResult['values'][0][0])

"""Performing and saving results for Polynomial Regression"""

# POLYNOMIAL REGRESSION
polynomialResult = predictPolynomialRegression(X_train, y_train, X_test, y_test)
polynomial_regressor = polynomialResult['regressor']
regValues['Polynomial'] = round(polynomialResult['values'][0][0])

"""Performing and saving results for Simple Vector Regression (SVR)"""

# SIMPLE VECTOR REGRESSION (SVR)
svrResult = predictSVR(X_train, y_train, X_test, y_test)
svr_regressor = svrResult['regressor']
regValues['SVR'] = round(svrResult['values'][0])

"""Voting Regressor"""

vRegressor = VotingRegressor(estimators=[('reg', reg), ('polynomial_regressor', polynomial_regressor), ('svr_regressor', svr_regressor)])

vRegressorRes = vRegressor.fit(X_train, y_train.ravel())

# VotingRegressor - Predict for January 2021
vRegressor_predict = vRegressor.predict([[predictFor]])
regValues['Voting Regressor'] = round(vRegressor_predict[0])
print('Voting Regressor January 2021 predicted value: ' + str(round(vRegressor_predict[0])))
regResults.loc[product] = regValues

"""Displaying all results"""

print(regResults)