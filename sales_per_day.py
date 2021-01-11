import numpy as np
import pandas as pd
import os
import io
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
s = pd.read_csv('data/salesdaily.csv',encoding='utf-8').fillna(0)
d = pd.read_csv('data/salesdaily.csv')
date = d['datum']

ab = s['M01AB'].iloc[0:2106].values
ae = s['M01AE'].iloc[0:2106].values
ba = s['N02BA'].iloc[0:2106].values
be = s['N02BE'].iloc[0:2106].values
b = s['N05B'].iloc[0:2106].values
c = s['N05C'].iloc[0:2106].values
r3 = s['R03'].iloc[0:2106].values
r6 = s['R06'].iloc[0:2106].values



fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(111)
ax.set_title("Sales per day For Last 5 years")
ax.set_xlabel('Date')
ax.set_ylabel('Sales per Day')
#ax.set_xlim([0,5])
ax.plot(date,ab,label='M01AB')
leg = ax.legend()
plt.show()

fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(111)
ax.set_title("Sales per day For Last 5 years")
ax.set_xlabel('Date')
ax.set_ylabel('Sales per Day')
#ax.set_xlim([0,5])
ax.plot(date,ae,label='M01AE')
leg = ax.legend()
plt.show()


fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(111)
ax.set_title("Sales per day For Last 5 years")
ax.set_xlabel('Date')
ax.set_ylabel('Sales per Day')
#ax.set_xlim([0,5])
ax.plot(date,ba,label='N02BA')
leg = ax.legend()
plt.show()

fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(111)
ax.set_title("Sales per day For Last 5 years")
ax.set_xlabel('Date')
ax.set_ylabel('Sales per Day')
#ax.set_xlim([0,5])
ax.plot(date,be,label='N02BE')
leg = ax.legend()
plt.show()

fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(111)
ax.set_title("Sales per day For Last 5 years")
ax.set_xlabel('Date')
ax.set_ylabel('Sales per Day')
#ax.set_xlim([0,5])
ax.plot(date,b,label='N05B')
leg = ax.legend()
plt.show()

fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(111)
ax.set_title("Sales per day For Last 5 years")
ax.set_xlabel('Date')
ax.set_ylabel('Sales per Day')
#ax.set_xlim([0,5])
ax.plot(date,c,label='N05C')
leg = ax.legend()
plt.show()

fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(111)
ax.set_title("Sales per day For Last 5 years")
ax.set_xlabel('Date')
ax.set_ylabel('Sales per Day')
#ax.set_xlim([0,5])
ax.plot(date,r3,label='R03')
leg = ax.legend()
plt.show()

fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(111)
ax.set_title("Sales per day For Last 5 years")
ax.set_xlabel('Date')
ax.set_ylabel('Sales per Day')
#ax.set_xlim([0,5])
ax.plot(date,r6,label='R06')
leg = ax.legend()
plt.show()