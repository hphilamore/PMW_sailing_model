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



# four_bit_sail_angles = pd.read_csv('actuator_data.csv')['end_to_end_angle']
four_bit_sail_angles = np.hstack((np.array(pd.read_csv('actuator_data.csv')['end_to_end_angle']),
	                              np.array(pd.read_csv('actuator_data.csv')['end_to_end_angle']) + pi))



def systematic_mode(num_points=10, binary=False):
	"""
	Systematically cycle through combination of each listed:
		- wind direction
		- wind speed
		- sail angle (option to auto adjust to wind angle within program)
		- rudder angle 
	(Sail angle used as starting angle where sail angle is auto-adjuested in program)
	"""
	# wind directions
	true_wind_dirs = [0, pi/6, pi/3, pi/2, pi*2/3, pi*5/6, pi, pi+pi/6, pi+pi/3, pi+pi/2, pi+pi*2/3, pi+pi*5/6, 2*pi]
	true_wind_speed = [5]
	sail_angles = [0]
	rudder_angles = [pi/8]
	#sail_angles = [0, pi/6, pi/3, pi/2, pi*2/3]

	T = np.arange(num_points)
	timestep = 1
	T_ticks = (T[0], T[-1], timestep)

	for twd in true_wind_dirs:
		print('WIND DIR =', twd)
		for tws in true_wind_speed:
			for r in rudder_angles:
				for s in sail_angles:
					main(rudder_angle = r, 
					     sail_angle = s,
					     auto_adjust_sail = True,
					     Time = T,
					     time_ticks = T_ticks,
					     true_wind_polar = [np.array([twd, tws])] * num_points,
					     binary_actuator = binary,
					     binary_angles = four_bit_sail_angles,
					     save_figs = False,#True,
					     fig_location = save_location,
					     plot_force_coefficients = False,
					     output_plot_title = f'twd:{twd}, tws:{tws}, r:{r}, s:{s}, , binary: {binary}')



def random_mode(num_points=5, binary=True):
	"""
	String of randomly generated wind values
	Systematically cycle through combination of each listed:
		- sail angle (option to auto adjust to wind angle within program)
		- rudder angle 
	(Sail angle used as starting angle where sail angle is auto-adjuested in program)
	"""
	sail_angles = [0]
	rudder_angles = [pi/8]

	angles = [random.uniform(0, 2*pi) for i in range(num_points)]
	mags = [random.normalvariate(5, 2) for i in range(num_points)]
	# angles = np.random.uniform(0, 2*pi, size=(num_points))
	# mags = np.random.normal(5, 2, size=(num_points))
	twp = [np.array([a, m]) for a, m in zip(angles, mags)]
	T = np.arange(num_points)
	timestep = 1
	T_ticks = (T[0], T[-1], timestep)

	for r in rudder_angles:
		for s in sail_angles:
			main(rudder_angle = r, 
			     sail_angle = s,
			     auto_adjust_sail = True,
			     Time = T,
			     time_ticks = T_ticks,
			     true_wind_polar = twp,
			     binary_actuator = binary,
			     binary_angles = four_bit_sail_angles,
			     save_figs = True,
			     fig_location = save_location,
			     plot_force_coefficients = False,
			     output_plot_title = 'random, binary: {binary}')



# empirically recorded wind data
wdID = 'PaddyA'
wdID = 'PaddyB'
#wdID = 'hillside'
#wdID = 'streamside'

def empirical_mode(wdID, data_points=slice(20,30), binary=True):
	"""
	String of empirically generated wind values
	Systematically cycle through combination of each listed:
		- sail angle (option to auto adjust to wind angle within program)
		- rudder angle 
	(Sail angle used as starting angle where sail angle is auto-adjuested in program)
	"""

	sail_angles = [0]
	rudder_angles = [0]

	# weather_data_streamside = pd.read_csv('weather_data_streamside_02-03-18_19-55_112cm.TXT', sep='\t')[['windspeed(m/s)', 'windAngle(deg)']][725:1000]
	# weather_data_hillside = pd.read_csv('weather_data_hillside_02-03-18_19-55.TXT', sep='\t')[['windspeed(m/s)', 'windAngle(deg)']][725:1000]
	# weather_data_paddyA = pd.read_csv('weather_data_paddy1_17-02-18_19-45_45cm.TXT', sep='\t')[['windspeed(m/s)', 'windAngle(deg)']]#[0:10]
	# weather_data_paddyB = pd.read_csv('weather_data_paddy1_17-02-18_18-45_45cm.TXT', sep='\t')[['windspeed(m/s)', 'windAngle(deg)']]#[0:10]

	if wdID == 'PaddyA':
		weather_data = pd.read_csv('weather_data_paddy1_17-02-18_19-45_45cm.TXT', sep='\t')[['windspeed(m/s)', 'windAngle(deg)']]#[0:10]# weather_data_paddyA
	elif wdID == 'PaddyB':
		weather_data = pd.read_csv('weather_data_paddy1_17-02-18_18-45_45cm.TXT', sep='\t')[['windspeed(m/s)', 'windAngle(deg)']]#[0:10]#weather_data_paddyB
	elif wdID == 'hillside':
		weather_data = pd.read_csv('weather_data_hillside_02-03-18_19-55.TXT', sep='\t')[['windspeed(m/s)', 'windAngle(deg)']][725:1000]# weather_data_hillside
	elif wdID == 'streamside':
		weather_data = pd.read_csv('weather_data_streamside_02-03-18_19-55_112cm.TXT', sep='\t')[['windspeed(m/s)', 'windAngle(deg)']][725:1000]# weather_data_streamside

	T, twp, T_ticks = empirical_data(weather_data, timestep=2, noise_sd=0.1, dp=data_points)

	for r in rudder_angles:
		for s in sail_angles:
			main(rudder_angle = r, 
			     sail_angle = s,
			     auto_adjust_sail = True,
			     Time = T,
			     time_ticks = T_ticks,
			     true_wind_polar = twp,
			     binary_actuator = binary,
			     binary_angles = four_bit_sail_angles,
			     save_figs = True,
			     fig_location = save_location,
			     plot_force_coefficients = False,
			     output_plot_title = f'start: {T_ticks[0]}, end: {T_ticks[1]}, timestep: {T_ticks[2]}, weather data: {wdID}, binary: {binary}')


# ### TEST CASES ###
# def wind_data_repeated_per_timestep(angle=pi - (pi/6), mag=5):
# 	"""
# 	Fixed wind angle and magnitude
# 	Wind data points repeated to match number of timesteps
# 	"""
# 	steps = 10
# 	T = np.arange(steps)
# 	twp = [np.array([angle, mag])] * steps
# 	return T, twp




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


# def random_wind_data(num_points=10):
# 	"""
# 	Randomly created wind data.
# 	Timesteps generated to match number of points
# 	"""
# 	angles = [random.uniform(0, 2*pi) for i in range(num_points)]
# 	mags = [random.normalvariate(5, 2) for i in range(num_points)]
# 	# angles = np.random.uniform(0, 2*pi, size=(num_points))
# 	# mags = np.random.normal(5, 2, size=(num_points))
# 	twp = [np.array([a, m]) for a, m in zip(angles, mags)]
# 	T = np.arange(len(twp))
# 	timestep = 1
# 	T_ticks = (T[0], T[-1], timestep)
# 	for r in rudder_angles:
# 		for s in sail_angles:
# 			main(rudder_angle = s, 
# 			     sail_angle = r,
# 			     auto_adjust_sail = True,
# 			     Time = T,
# 			     time_ticks = T_ticks,
# 			     true_wind_polar = twp,
# 			     binary_actuator = False,
# 			     binary_angles = four_bit_sail_angles,
# 			     save_figs = True,
# 			     fig_location = save_location,
# 			     plot_force_coefficients = False,
# 			     weather_data_ID = wdID)


# def cycle_wind_data():
# 	"""
# 	Systematially cycle through each wind speed and magnitude given in two lists
# 	Timesteps generated to match number of points
# 	"""
# 	twp = [np.array([twd, tws]) for twd in true_wind_dirs for tws in true_wind_speed]
# 	T = np.arange(len(twp))
# 	timestep = 1
# 	T_ticks = (T[0], T[-1], timestep)
# 	return T, twp, T_ticks


def empirical_data(df, timestep=2, noise_sd=0.1, dp=slice(20,30)):
	"""
	Uses data frame constraucted from input empirical data to create an array of true wind values.
	Interpolated points and noise as optional inputs.

	Inputs
	df : the data frame
	freq : entries per minute (raw data = 1)
	noise : 
	"""

	if timestep < 1:
		raise ValueError("Timestep < 1 entered. Minimum timestep is 1 second.")

	windSpeed = df['windspeed(m/s)']
	windAngle = np.deg2rad(df['windAngle(deg)'])

	# original time points, converted mins to s
	time = np.arange(len(df)) * 60
	print('time', time)
	print(len(time))
	
	# time points for interpolated data set
	num_points = len(time) * (60/timestep)
	time_int = np.arange(time[0], time[-1]+1, timestep)
	print('time_int', time_int)
	print(len(time_int))
	
	# order of interpolation data
	order_poly = 3

	# interpolation and added noise data
	func = splrep(time, windSpeed, k=order_poly)
	windSpeed_int = splev(time_int, func)
	windSpeed_noisy = windSpeed_int + np.random.normal(0, noise_sd, size=windSpeed_int.shape)

	func = splrep(time, windAngle, k=order_poly)
	windAngle_int = splev(time_int, func)
	windAngle_noisy = windAngle_int + pi/2 * np.random.normal(0, noise_sd, size=windAngle_int.shape) # pi/2 * np.random.random(size=windAngle_int.shape)

	# plot data to check
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

	# organise into polar coordinates
	twp = [np.array([twd, tws]) for twd, tws in zip(windAngle_noisy, windSpeed_noisy)]

	# select a slice of the data
	twp = twp[dp]
	T = time_int[dp]

	# save the limits and timestep used
	T_ticks = np.arange(T[0], T[-1]+1, timestep)
	print(T_ticks)

	# integer series, length = number of time points
	T = np.arange(len(T))

	# plot to check
	fig1, ax1 = plt.subplots()	
	twp_ = np.stack(twp)
	# print(twp_)
	plt.plot(T, twp_[:, 0], label=f'angle')
	plt.plot(T, twp_[:, 1], label=f'speed')
	plt.xlabel('time (secs')
	plt.xticks(T, T_ticks)
	plt.legend()
	plt.savefig(f'{save_location}/wind_profile.pdf')
	plt.show()

	return T, twp, T_ticks


# # T, twp, T_ticks = cycle_wind_data()
# T, twp, T_ticks = random_wind_data(num_points=5)
# T, twp, T_ticks = empirical_data(weather_data, timestep=2, noise_sd=0.1, dp=slice(20,30))


# for r in rudder_angles:
# 	for s in sail_angles:
# 		main(rudder_angle = s, 
# 		     sail_angle = r,
# 		     auto_adjust_sail = True,
# 		     Time = T,
# 		     time_ticks = T_ticks,
# 		     true_wind_polar = twp,
# 		     binary_actuator = False,
# 		     binary_angles = four_bit_sail_angles,
# 		     save_figs = True,
# 		     fig_location = save_location,
# 		     plot_force_coefficients = False,
# 		     weather_data_ID = wdID)
#empirical_mode('PaddyB', binary=False)

#random_mode(binary=True)

systematic_mode(binary=True)






