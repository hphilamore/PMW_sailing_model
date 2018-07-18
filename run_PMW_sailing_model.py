from PMW_sailing_model import *
import numpy as np
from numpy import pi, sin, cos, rad2deg, deg2rad, arctan2, sqrt, exp
import os, time, fnmatch
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splrep
from scipy.interpolate import splev
import scipy.interpolate
import matplotlib.cm as cm


# set up a folder to store data
for root, dirs, files in os.walk("/Users/hemma/Documents/Projects"):
		for d in dirs:    
			if fnmatch.fnmatch(d, "sailing_simulation_results"):
				save_location = os.path.join(root, d) + '/' + time.strftime('%Y-%m-%d--%H-%M-%S')
				os.makedirs(save_location, exist_ok=True)



# four_bit_sail_angles = pd.read_csv('actuator_data.csv')['end_to_end_angle']
four_bit_sail_angles = np.hstack((np.array(pd.read_csv('actuator_data.csv')['end_to_end_angle']),
	                              np.array(pd.read_csv('actuator_data.csv')['end_to_end_angle']) + pi))





def systematic(num_points=20, timestep=1):
	true_wind_dirs = [0, pi/6, pi/3, pi/2, pi*2/3, pi*5/6, pi, pi+pi/6, pi+pi/3, pi+pi/2, pi+pi*2/3, pi+pi*5/6, 2*pi]
	# true_wind_dirs = [0, pi/4, pi/2, pi*3/4, pi, pi+pi/4, pi+pi/2, pi+pi*3/4, 2*pi]
	# true_wind_dirs = [pi+pi/2]#, pi+pi/2, pi+pi*3/4, 2*pi]
	true_wind_speed = [5]

	steps = np.arange(num_points)
	time_ticks = (steps[0], steps[-1], timestep) # tuple input to graph plot ticks function

	colours = cm.rainbow(np.linspace(0, 1, len(true_wind_dirs)))

	fig, ax = plt.subplots()

	for tws in true_wind_speed:
		for twd, c in zip(true_wind_dirs, colours):
			true_wind_polar = [np.array([twd, tws])] * num_points
			print(true_wind_polar)
			run_model(true_wind_polar, 
				steps, 
				time_ticks, 
				ax, 
				c)

	
	ax.legend()
	plt.show()
			

	#title = f'binary: {b}, latency: {l}s, twd: {true_wind_polar[0]}rads, tws:{true_wind_polar[1]}m/s, rud ang:{r}rads, sail ang:{s}rads'


		




def run_model(twp,
			  steps,
			  timeticks,	
			  axes,
			  line_colour = 'k',
			  latency = [0],#, 2, 4, 6],
			  binary = [True, False],
			  rudder_angles = [pi/8],
			  sail_angles = [0],
			  ):# , binary=False, Latency=0):
	"""
	Systematically cycle through combination of each listed:
		- wind direction
		- wind speed
		- sail angle (option to auto adjust to wind angle within program)
		- rudder angle 
	(Sail angle used as starting angle where sail angle is auto-adjuested in program)
	"""


	# set up output figure
	lines = ['-', ':']
	markers = ["o", "<",  "2", "3", "v", "4", "8", "s", "p", "P", "^", "*", "1","h", "H", "+", "x", "X", "D", ">",  "d", "|", "_"]
	
	# fig, ax = plt.subplots()
	# fig2, ax2 = plt.subplots()

	data = []


	for b, li in binary, lines:
		for l, ma in zip(latency, markers):
			for r in rudder_angles:
				for s in sail_angles:
					d = main(rudder_angle = r, 
					     sail_angle = s,
					     auto_adjust_sail = True,
					     Time = steps,
					     time_ticks = timeticks,
					     true_wind_polar = twp, #[np.array([twd, tws])] * num_points,
					     binary_actuator = b,
					     binary_angles = four_bit_sail_angles,
					     draw_boat_plot = False,
					     save_figs = False,#True,
					     show_figs = False, #True,
					     fig_location = save_location,
					     plot_force_coefficients = False,
					     #output_plot_title = plot_title,#f'binary: {b}, latency: {l}s, twd: {true_wind_polar[0]}rads, tws:{true_wind_polar[1]}m/s, rud ang:{r}rads, sail ang:{s}rads' ,
					     latency = l)

					positions = np.vstack([pol2cart(p) for p in d['position']])
					print([x[0] for x in enumerate(d['heading']) if abs(x[1]) > pi])
					full_turn = next((x[0] for x in enumerate(d['heading']) if abs(x[1]) > pi), len(d['heading']))
					print('full_turn', full_turn)
					positions = positions[: full_turn]
					#ax.plot(positions[:,0], positions[:,1], '-o')	
					#axes.plot(positions[:,0], positions[:,1], linestyle=li, color=line_colour, marker=ma)	
					axes.plot(positions[:,0], positions[:,1], marker=ma, label=str(round(twp[0][0],3)))	

	#plt.show()


def systematic_mode(num_points=20):# , binary=False, Latency=0):
	"""
	Systematically cycle through combination of each listed:
		- wind direction
		- wind speed
		- sail angle (option to auto adjust to wind angle within program)
		- rudder angle 
	(Sail angle used as starting angle where sail angle is auto-adjuested in program)
	"""
	# wind directions
	#binary = [False, True]#[False, True]
	latency = [2, 5, 10]
	true_wind_dirs = [0, pi/6, pi/3, pi/2, pi*2/3, pi*5/6, pi, pi+pi/6, pi+pi/3, pi+pi/2, pi+pi*2/3, pi+pi*5/6, 2*pi]
	true_wind_dirs = [0, pi/4, pi/2, pi*3/4, pi, pi+pi/4, pi+pi/2, pi+pi*3/4, 2*pi]
	true_wind_dirs = [pi+pi/2]#, pi+pi/2, pi+pi*3/4, 2*pi]
	true_wind_speed = [5]
	rudder_angles = [pi/8]
	sail_angles = [0]
	#sail_angles = [0, pi/6, pi/3, pi/2, pi*2/3]

	# plot features
	lines = ['-', ':']
	markers = ["o",   "<",  "2", "3", "v", "4", "8", "s", "p", "P", "^", "*", "1","h", "H", "+", "x", "X", "D", ">",  "d", "|", "_"]
	colours = cm.rainbow(np.linspace(0, 1, len(true_wind_dirs)))


	T = np.arange(num_points)
	timestep = 1
	T_ticks = (T[0], T[-1], timestep)
	data = []
	fig, ax = plt.subplots()
	fig2, ax2 = plt.subplots()

	binary=[True, False]


	# for b, l in zip(B, L):
	for n, (b, li) in enumerate(zip(binary, lines)):
		print(b)
		for l, ma in zip(latency, markers):
			for twd, co in zip(true_wind_dirs, colours):
				print('WIND DIR =', twd)
				for tws in true_wind_speed:
					for r in rudder_angles:
						for s in sail_angles:
							print('binary=', n)
							d = main(rudder_angle = r, 
							     sail_angle = s,
							     auto_adjust_sail = True,
							     Time = T,
							     time_ticks = T_ticks,
							     true_wind_polar = [np.array([twd, tws])] * num_points,
							     binary_actuator = b,
							     binary_angles = four_bit_sail_angles,
							     draw_boat_plot = False,
							     save_figs = False,#True,
							     show_figs = False, #True,
							     fig_location = save_location,
							     plot_force_coefficients = False,
							     output_plot_title = f'binary: {b}, latency: {l}s, twd: {twd}rads, tws:{tws}m/s, rud ang:{r}rads, sail ang:{s}rads' ,
							     latency = l)

							
							# print(np.vstack(d['position'])[:,0])
							# print(np.vstack(d['position'])[:,1])
							# print(np.vstack(d['position']))
							#print('heading')
							#print(d['heading'])
							positions = np.vstack([pol2cart(p) for p in d['position']])
							print([x[0] for x in enumerate(d['heading']) if abs(x[1]) > pi])
							#print(len([p for p in d['heading'] if abs(p)<pi]))
								# TODO this is more elegent but not sure how it is working
							full_turn = next((x[0] for x in enumerate(d['heading']) if abs(x[1]) > pi), len(d['heading']))
							
							#full_turn = next((x[0] for x in enumerate(d['heading']) if abs(x[1]) > pi)) 
							# if (full_turn == len(d['heading']) - 1):
							# 	full_turn = 0
							# print(next((i for i in range(10) if i**2 == 17), None))
							# except:
							# 	print('timeout')
							# 	full_turn = 0
							print('full_turn', full_turn)
							positions = positions[: full_turn]
							ax.plot(positions[:,0], positions[:,1], '-o', label=str(round(twd,3)))	
							#ax2.plot(positions[:,0], positions[:,1], 'o', label=str(twd))	
							ax2.plot(positions[:,0], positions[:,1], linestyle=li, color=co, marker=ma, label=str(twd))	

							#full_turn = next(x[0] for x in enumerate(d['heading']) if x[1] > pi)

					

							# ax.plot(np.vstack(d['position'])[:,1], np.vstack(d['position'])[:,0], '-o', label=str(twd))	
							# ax2.plot(np.vstack(d['position'])[:,1], np.vstack(d['position'])[:,0], 'o', label=str(twd))	
							#data.append(d)
							#plt.plot(np.vstack(data['position'][0], np.vstack(data['position'][1])))
		#ax2.legend()
	plt.show()


	
	#print(data)

def random_mode(num_points=50, binary=True, Latency=0):
	"""
	String of randomly generated wind values
	Systematically cycle through combination of each listed:
		- sail angle (option to auto adjust to wind angle within program)
		- rudder angle 
	(Sail angle used as starting angle where sail angle is auto-adjuested in program)
	"""
	sail_angles = [0]
	rudder_angles = [pi/4]

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
			     draw_boat_plot = True,
			     save_figs = True,
			     show_figs = False,
			     fig_location = save_location,
			     plot_force_coefficients = False,
			     output_plot_title = 'random, binary: {binary}',
			     latency = Latency)



# empirically recorded wind data

#wdID = 'PaddyB'
#wdID = 'hillside'
#wdID = 'streamside'

def empirical_mode(wdID= 'PaddyA', data_points=slice(20,30), binary=True):
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
			     draw_boat_plot = True,
			     save_figs = True,
			     show_figs = False,
			     fig_location = save_location,
			     plot_force_coefficients = False,
			     output_plot_title = f'start: {T_ticks[0]}, end: {T_ticks[1]}, timestep: {T_ticks[2]}, weather data: {wdID}, binary: {binary}',
			     latency = Latency)


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




def curvature(points):
	"""
	https://stackoverflow.com/questions/28269379/curve-curvature-in-numpy
	"""
	dx_dt = np.gradient(a[:, 0])
	dy_dt = np.gradient(a[:, 1])
	velocity = np.array([ [dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])
	ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
	tangent = np.array([1/ds_dt] * 2).transpose() * velocity
	tangent_x = tangent[:, 0]
	tangent_y = tangent[:, 1]
	deriv_tangent_x = np.gradient(tangent_x)
	deriv_tangent_y = np.gradient(tangent_y)
	dT_dt = np.array([ [deriv_tangent_x[i], deriv_tangent_y[i]] for i in range(deriv_tangent_x.size)])
	length_dT_dt = np.sqrt(deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y)
	normal = np.array([1/length_dT_dt] * 2).transpose() * dT_dt
	d2s_dt2 = np.gradient(ds_dt)
	d2x_dt2 = np.gradient(dx_dt)
	d2y_dt2 = np.gradient(dy_dt)
	curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
	t_component = np.array([d2s_dt2] * 2).transpose()
	n_component = np.array([curvature * ds_dt * ds_dt] * 2).transpose()
	acceleration = t_component * tangent + n_component * normal
	return curvature

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

# B = [False, True]
# L = [0, 5, 10]

# for b, l in zip(B, L):
#systematic_mode(binary=False)#, Latency=l)

#systematic_mode()#, Latency=l)
systematic()





