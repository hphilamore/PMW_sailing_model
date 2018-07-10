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

# set up a folder to store data
for root, dirs, files in os.walk("/Users/hemma/Documents/Projects"):
		for d in dirs:    
			if fnmatch.fnmatch(d, "sailing_simulation_results"):
				save_location = os.path.join(root, d) + '/' + time.strftime('%Y-%m-%d--%H-%M-%S')
				os.makedirs(save_location, exist_ok=True)



# wind directions
true_wind_dirs = [0, pi/6, pi/3, pi/2, pi*2/3, pi*5/6, pi, pi+pi/6, pi+pi/3, pi+pi/2, pi+pi*2/3, pi+pi*5/6, 2*pi]
true_wind_speed = [5]

# initial sail and rudder angles as list of cases
sail_angles = [0]
rudder_angles = [0]
#sail_angles = [0, pi/6, pi/3, pi/2, pi*2/3]


# four_bit_sail_angles = pd.read_csv('actuator_data.csv')['end_to_end_angle']
four_bit_sail_angles = np.hstack((np.array(pd.read_csv('actuator_data.csv')['end_to_end_angle']),
	                              np.array(pd.read_csv('actuator_data.csv')['end_to_end_angle']) + pi))



# empirically recorded wind data
weather_data_streamside = pd.read_csv('weather_data_streamside_02-03-18_19-55_112cm.TXT', sep='\t')[['windspeed(m/s)' , 'windAngle(deg)']][725:1000]
weather_data_hillside = pd.read_csv('weather_data_hillside_02-03-18_19-55.TXT', sep='\t')[['windspeed(m/s)' , 'windAngle(deg)']][725:1000]
#weather_data_streamside = weather_data_streamside[0:10]
#weather_data_hillside = weather_data_hillside[0:10]
weather_data_paddyA = pd.read_csv('weather_data_paddy1_17-02-18_19-45_45cm.TXT', sep='\t')[['windspeed(m/s)' , 'windAngle(deg)']]#[0:10]
weather_data_paddyB = pd.read_csv('weather_data_paddy1_17-02-18_18-45_45cm.TXT', sep='\t')[['windspeed(m/s)' , 'windAngle(deg)']]#[0:10]


weather_data = weather_data_paddyA
weather_data = weather_data_paddyB
weather_data = weather_data_hillside
weather_data = weather_data_streamside




### TEST CASES ###
def wind_data_repeated_per_timestep(angle=pi - (pi/6), mag=5):
	"""
	Fixed wind angle and magnitude
	Wind data points repeated to match number of timesteps
	"""
	steps = 10
	T = np.arange(steps)
	twp = [np.array([angle, mag])] * steps
	return T, twp


# def random_wind_data():
# 	"""
# 	Randomly created wind data.
# 	Timesteps generated to match number of points
# 	"""
# 	random_wind_dirs = true_wind_dirs
# 	random.shuffle(random_wind_dirs)
# 	twp = [np.array([r, 5]) for r in random_wind_dirs]
# 	T = np.arange(len(twp))
# 	return T, twp


def random_wind_data(num_points=10):
	"""
	Randomly created wind data.
	Timesteps generated to match number of points
	"""
	angles = [random.uniform(0, 2*pi) for i in range(num_points)]
	mags = [random.normalvariate(5, 2) for i in range(num_points)]
	# angles = np.random.uniform(0, 2*pi, size=(num_points))
	# mags = np.random.normal(5, 2, size=(num_points))
	twp = [np.array([a, m]) for a, m in zip(angles, mags)]
	T = np.arange(len(twp))
	return T, twp, 1


def cycle_wind_data():
	"""
	Systematially cycle through each wind speed and magnitude given in two lists
	Timesteps generated to match number of points
	"""
	twp = [np.array([twd, tws]) for twd in true_wind_dirs for tws in true_wind_speed]
	T = np.arange(len(twp))
	return T, twp, 1

def empirical_data(df, freq=1, noise_sd=0.1, dp=slice(20,30)):
	"""
	Uses data frame constraucted from input empirical data to create an array of true wind values.
	Interpolated points and noise as optional inputs.

	Inputs
	df : the data frame
	freq : entries per minute (raw data = 1)
	noise : 
	"""
	#entries = len(df)#['windspeed(m/s)'])

	windSpeed = df['windspeed(m/s)']
	windAngle = np.deg2rad(df['windAngle(deg)'])

	# original time points, converted to (s)
	time = np.linspace(0, len(df), len(df)) * 60
	
	# time points for interpolated data set
	points = len(df) * freq
	time_int = np.linspace(0, len(df), points)
	

	# interpolated values
	order_poly = 3

	func = splrep(time, windSpeed, k=order_poly)
	windSpeed_int = splev(time_int, func)
	windSpeed_noisy = windSpeed_int + np.random.normal(0, noise_sd, size=windSpeed_int.shape)

	func = splrep(time, windAngle, k=order_poly)
	windAngle_int = splev(time_int, func)
	windAngle_noisy = windAngle_int + pi/2 * np.random.normal(0, noise_sd, size=windAngle_int.shape) # pi/2 * np.random.random(size=windAngle_int.shape)

	# fig1, ax1 = plt.subplots()	
	# plt.plot(time_int, windSpeed_noisy, label=f'speed noisy')
	# plt.plot(time_int, windSpeed_int, '--', label=f'speed interp')
	# plt.plot(time, windSpeed, label=f'speed raw')
	# plt.xlabel('time (mins')
	# plt.ylabel('wind speed (m/s)')
	# plt.legend()

	# fig1, ax1 = plt.subplots()	
	# plt.plot(time_int, windAngle_noisy, label=f'angle noisy')
	# plt.plot(time_int, windAngle_int, '--', label=f'angle interp')
	# plt.plot(time, windAngle, label=f'angle raw')
	# plt.xlabel('time (mins')
	# plt.ylabel('wind angle (rads)')
	# plt.legend()
	#plt.show()


	twp = [np.array([twd, tws]) for twd, tws in zip(windAngle_noisy, windSpeed_noisy)]

	# # integer series, length = number of time points
	# #T = time_int
	# T = np.arange(0, len(time_int))

	# timestep (s)
	ts = 60 / freq

	# select a slice of the data
	twp = twp[dp]
	T = time_int[dp]

	print(twp)
	print(T)
	print()

	T_ticks = (min(T), max(T), ts)

	


	# integer series, length = number of time points
	#T = time_int
	T = np.arange(len(T))

	print(twp)
	print(T)



	fig1, ax1 = plt.subplots()	
	twp_ = np.stack(twp)
	print(twp_)
	plt.plot(T*ts, twp_[:, 0], label=f'angle')
	plt.plot(T*ts, twp_[:, 1], label=f'speed')
	plt.xlabel('time (secs')
	plt.legend()
	plt.show()

	# select a section of the total time collected to display
	# max_timestep = out_mins * freq
	# twp = twp[ : max_timestep]
	# T = T[ : max_timestep]


	# print('T', T)
	# print('twp', twp)

	return T, twp, T_ticks







T, twp, ts = cycle_wind_data()
T, twp, ts = random_wind_data()
T, twp, ts = empirical_data(weather_data, freq=30, noise_sd=0.1)

# T = T[0:10]
# twp = twp[0:10]

print(T, twp)



for r in rudder_angles:
	for s in sail_angles:
		main(rudder_angle = s, 
		     sail_angle = r,
		     auto_adjust_sail = True,
		     Time = T,
		     timestep = ts,
		     true_wind_polar = twp,
		     binary_actuator = False,
		     binary_angles = four_bit_sail_angles,
		     save_figs = True,
		     fig_location = save_location,
		     plot_force_coefficients = False)



