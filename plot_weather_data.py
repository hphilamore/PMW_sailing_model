from PMW_sailing_model import *
import numpy as np
from numpy import pi
import os, time, fnmatch
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splrep
from scipy.interpolate import splev
import scipy.interpolate


def pol2cart(coords):
	"""
	Converts polar to cartesian coordinates 
	Returns x, y
	"""
	phi = coords[0]
	rho = coords[1]
	x = rho * cos(phi)
	y = rho * sin(phi)
	#return(x, y)
	np.array([x,y])
	return np.array([x, y])


weather_data = {}
weather_data['Paddy A'] = pd.read_csv('weather_data_paddy1_17-02-18_19-45_45cm.TXT', sep='\t')[['windspeed(m/s)', 'windAngle(deg)']]#[0:10]# weather_data_paddyA
weather_data['Paddy B'] = pd.read_csv('weather_data_paddy1_17-02-18_18-45_45cm.TXT', sep='\t')[['windspeed(m/s)', 'windAngle(deg)']]#[0:10]#weather_data_paddyB
weather_data['Hill-side'] = pd.read_csv('weather_data_hillside_02-03-18_19-55.TXT', sep='\t')[['windspeed(m/s)', 'windAngle(deg)']][745:860]# weather_data_hillside
weather_data['Stream-side'] = pd.read_csv('weather_data_streamside_02-03-18_19-55_112cm.TXT', sep='\t')[['windspeed(m/s)', 'windAngle(deg)']][840:960]# weather_data_streamside



for i in weather_data:
	#select first 50mins of data
	weather_data[i] = weather_data[i][0:50]
	df = weather_data[i]
	#print(df, weather_data[df])
	windSpeed = df['windspeed(m/s)']
	windAngle = np.deg2rad(df['windAngle(deg)'])
	# original time points, converted mins to s
	time = np.arange(len(df))# * 60
	print('time', time)
	print(len(time))
	
	# plot data to check
	fig1, ax1 = plt.subplots()	
	ax1.axhline(y=0, color='k', linewidth=1.0, alpha=0.5)
	ax1.axvline(x=0, color='k', linewidth=1.0)
	plt.plot(time, windSpeed, label=f'Velocity')
	plt.xlabel('time (mins')
	plt.ylabel('Wind Velocity (m/s)')
	plt.xlim([0, 50])
	plt.ylim([-0.4, 4.5])

	posx, posy = 0.8, 0.9
	if i == 'Paddy B':
		posy = 0.1

	

	for t, A, S in zip(time, windAngle, windSpeed):
		quiver_scale = 20
		angle = pol2cart(np.array([A, 1]))
		Q = plt.quiver(t, S, angle[0], angle[1], scale=quiver_scale)
		#plt.quiverkey(Q, -1.5, n/2-2, 0.25, label, coordinates='data')
		quiver_key_scale = quiver_scale/10#100
		plt.quiverkey(Q, posx, posy, quiver_key_scale, 'Wind Angle', coordinates='axes', labelpos='E')

	plt.title(i)
	#plt.legend(frameon=False)

plt.show()
