from PMW_sailing_model import *
from empirical_weather_data import *
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
import itertools

'''
Model of a sailing boat (or portuguese man of war jellyfish).


One of three function calls will runthe model:
	- systematic() : cycles through incremental combinations of sail and wind angle
	- random wind() : wind speed and direction are generated randomly
	- empirical() : wind speed and direction from field data.


Function run_model has the following user definable parameters:
[ In each case, if a list is given the program will run each possible combination of values from all lists ]
	latency  			: timesteps sail angle takes to react to wind direction 
	binary 				: True or False - sail has discrete set of possible angular positions
	rudder_angles 		: Angle of rudder throughout the 
	sail_angles 		: (starting) Angle of sail
	auto_adjust_sail	: Sail position responds to incoming wind angle


Returns data at each timestep as a data structure, d.

The boat position at each timestep is plotted. 


TODO: Rewrite as object oriented model where eah run mode is a different sub-class 
TODO: Use ODE solver so that force doesn't get huge
TODO: add tacking (rudder angle is a function of postion)


'''

# set up a folder to store data
for root, dirs, files in os.walk("/Users/hemma/Documents/Projects"):
		for d in dirs:    
			if fnmatch.fnmatch(d, "sailing_simulation_results"):
				save_location = os.path.join(root, d) + '/' + time.strftime('%Y-%m-%d--%H-%M-%S')
				os.makedirs(save_location, exist_ok=True)


fontsize = 12


# import a list of possible sail angles
# TODO: have the model of the actutor run to feedin these values rather than importing them from a data file
# four_bit_sail_angles = pd.read_csv('actuator_data.csv')['end_to_end_angle']
four_bit_sail_angles = np.hstack((np.array(pd.read_csv('actuator_data.csv')['end_to_end_angle']),
	                              np.array(pd.read_csv('actuator_data.csv')['end_to_end_angle']) + pi))


def four_quad(angle):
	"""
	Converts angle to 4 quadrants, positive angles 0 --> 2*pi
	Right hand coordinate system. 
	"""
	if angle > 2*pi:
		angle -= 2*pi

	elif angle < 0:
		angle += 2*pi

	return angle 



def systematic(time_points=30):
	'''
	Runs the model for all combinations of: 
		- wind speed in list
		- wind angle in list
	over number of time points given. 
	'''

	#true_wind_dirs = [0, pi/6, pi/3, pi/2, pi*2/3, pi*5/6, pi, pi+pi/6, pi+pi/3, pi+pi/2, pi+pi*2/3, pi+pi*5/6, 2*pi]
	true_wind_dirs = [pi*3/4, pi/2, pi/4, 0]
	# true_wind_dirs = [pi+pi/2]#, pi+pi/2, pi+pi*3/4, 2*pi]
	true_wind_speed = [5]

	Time = np.arange(time_points)

	colours = cm.cool(np.linspace(0, 1, len(true_wind_dirs)))

	fig, ax = plt.subplots()
	#fig_, ax_ = plt.subplots()

	for tws in true_wind_speed:
		for twd, c in zip(true_wind_dirs, colours):
			true_wind_polar = [np.array([twd, tws])] * time_points
			#print(true_wind_polar)
			run_model(true_wind_polar, 
					  Time, 
					  ax, 
					  c)
	#ax.legend(bbox_to_anchor=(1.05, 1), loc=2, frameon=False)
	#fig.savefig('samplefigure', bbox_extra_artists=(lgd,), bbox_inches='tight')
	handles, labels = ax.get_legend_handles_labels()
	lgd = ax.legend(handles, labels, bbox_to_anchor=(0.4, -0.1), frameon=False)
	plt.savefig(f'{save_location}/plot.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
	#ax.legend(frameon=False)
	plt.show()



def random_wind(time_points=20, binary=True):
	"""
	Generates a random combination of wind speed and angle at each time point
	"""

	angles = [random.uniform(0, 2*pi) for i in range(time_points)]
	mags = [random.normalvariate(5, 2) for i in range(time_points)]
	# angles = np.random.uniform(0, 2*pi, size=(time_points))
	# mags = np.random.normal(5, 2, size=(time_points))
	true_wind_polar = [np.array([a, m]) for a, m in zip(angles, mags)]
	Time = np.arange(time_points)

	fig, ax = plt.subplots()

	run_model(true_wind_polar, 
			  Time, 
			  ax)
			

	ax.legend()
	plt.show()
	#title = f'binary: {b}, latency: {l}s, twd: {true_wind_polar[0]}rads, tws:{true_wind_polar[1]}m/s, rud ang:{r}rads, sail ang:{s}rads'


def empirical(time_points=100, generated_points_per_sec=0.5):
	'''
	Wind speed and direction taken from a section of empirical data.
	Empirical data timestep = 2mins so data interolated to give generated_points_per_sec in plac of 1 point per 120sec (empirical data).
	Noise added using function shown below. 

	'''


	weather_data = import_weather_data()
	#Time = np.arange(len(weather_data)) * 60

	colours = cm.rainbow(np.linspace(0, 1, len(weather_data)))

	fig, ax = plt.subplots()
	#fig_, ax_ = plt.subplots()

	keys = list(weather_data.keys())[1:2]#[:-1]#[:-1]
	#print(keys)


	for key, c in zip(keys, colours):

		#print(key)

		df = weather_data[key]
		angle = list(df['windAngle(rad)'])
		speed = list(df['windspeed(m/s)'])
		angle, speed = interpolate_add_noise(angle, speed, generated_points_per_sec)

		# organise interpolated into polar coordinates
		true_wind_polar = [np.array([twd, tws]) for twd, tws in zip(angle, speed)]

		# if interpolated points have a frequency of <1, create mulitple points per s
		true_wind_polar = list(itertools.chain.from_iterable(itertools.repeat(twp, 3) for twp in true_wind_polar))[:time_points]

		true_wind_polar = np.vstack(true_wind_polar)
		#print(true_wind_polar)

		#print(true_wind_polar)

		Time = np.arange(time_points)

		# preview input data
		fig1, ax1 = plt.subplots()	
		plt.plot(Time, true_wind_polar[:, 0], label=f'angle')
		plt.plot(Time, true_wind_polar[:, 1], label=f'speed')
		plt.xlabel('time (secs')
		plt.legend()
		plt.savefig(f'{save_location}/wind_profile.pdf')
		plt.close()
		#plt.show()

		run_model(true_wind_polar, 
				  Time, 
				  ax, 
				  c)

	plt.show()




def interpolate_add_noise(angle, speed, points_per_sec, interpolation_order=3, noise_sd=0.1):
	"""
	Uses data frame constraucted from input empirical data to create an array of true wind values.
	Interpolated points and noise as optional inputs.

	Inputs
	df : the data frame
	freq : entries per minute (raw data = 1)
	noise : 
	"""

	if points_per_sec > 1:
		raise ValueError("Timestep < 1s entered. Minimum timestep is 1s second. Maximum sensor frequency.")

	#print(len(angle))
	time_orig = np.arange(len(angle)) * 60
	#print(time_orig)

	#time_points = len(Time) * 60 * points_per_sec
	time_int = np.arange( 0, time_orig[-1]+1, 1/points_per_sec ) #* 60

	order_poly = 3

	np.random.seed(9002)

	# interpolation and added noise data
	func = splrep(time_orig, speed, k=order_poly)
	windSpeed_int = splev(time_int, func)
	#np.random.seed(9002)
	windSpeed_noisy = windSpeed_int + np.random.normal(0, noise_sd, size=windSpeed_int.shape)

	func = splrep(time_orig, angle, k=order_poly)
	windAngle_int = splev(time_int, func)
	#np.random.seed(9002)
	windAngle_noisy = windAngle_int + pi/2 * np.random.normal(0, noise_sd, size=windAngle_int.shape) # pi/2 * np.random.random(size=windAngle_int.shape)
	

	# plot data to check
	fig1, ax1 = plt.subplots()	
	plt.plot(time_int, windSpeed_noisy, label=f'speed noisy')
	plt.plot(time_int, windSpeed_int, '--', label=f'speed interp')
	plt.plot(time_orig, speed, label=f'speed raw')
	plt.xlabel('time (mins')
	plt.ylabel('wind speed (m/s)')
	plt.legend()
	#plt.show()
	plt.close()

	fig1, ax1 = plt.subplots()	
	plt.plot(time_int, windAngle_noisy, label=f'angle noisy')
	plt.plot(time_int, windAngle_int, '--', label=f'angle interp')
	plt.plot(time_orig, angle, label=f'angle raw')
	plt.xlabel('time (mins')
	plt.ylabel('wind angle (rads)')
	plt.legend()
	#plt.show()
	plt.close()

	

	speed = windSpeed_noisy
	angle = windAngle_noisy

	return angle, speed




def run_model(twp,
			  Time_,
			  axes,
			  line_colour = 'k',
			  latency = [0, 2, 4, 6],
			  binary = [False, True],
			  rudder_angles = [pi/4],
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
	face_colours = [line_colour, 'none']


	# print('twp', twp)
	# angle = pol2cart(twp[0])
	# Q = axes.quiver(0.5, -1, angle[0], angle[1], color=line_colour, units='x', scale=10)
	# axes.annotate(r'$\theta_w$', xy=(0.5, -1), xytext=(0.45, -1.3), size=16)

	data = []

	for b, li, fc in zip(binary, lines, face_colours):
		for l, ma in zip(latency, markers):
			for r in rudder_angles:
				for s in sail_angles:
					d = main(rudder_angle = r, 
					     sail_angle = s,
					     auto_adjust_sail = True,
					     Time = Time_,
					     true_wind_polar = twp, #[np.array([twd, tws])] * time_points,
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

					#print([x[0] for x in enumerate(d['heading']) if abs(x[1]) > pi])

					# rudder is angled
					if r!=0:
						# plot posisions until boat reaches angular rotation of 'stopping_angle'
						stopping_angle = pi

						# number of positions up to but not including position at which boat reaches 'stopping angle'
						u_turn = next((x[0] for x in enumerate(d['heading']) if abs(x[1]) > stopping_angle), len(d['heading']))

						# select positions up to and including stopping angle
						positions = np.array(positions[: u_turn+1])
		
						# convert to np array
						#points = np.array(positions)	

						# find average curvature of the trajectory			
						curv = curvature(positions)

						# the heading when the boat reaches the stpping angle
						# final_heading = d['heading'][u_turn-1]
						final_heading = d['heading'][u_turn]

						# divide stopping angle by no of positions (no. of seconds) to make the turn ---> average angular velocity
						ave_ang_vel = (final_heading / u_turn)
						# divide curvature by no of positions (no. of seconds) to make the turn ---> average curvature
						ave_curvature = (np.sum(curv) / u_turn)


						# string with info of average turning velocity
						#info = ', ' + 'r$\-\.\lambda=$' + str(round(ave_ang_vel, 2))
						info = ',  ' + r'$\bar\omega=$' + str(abs(round(ave_ang_vel, 2))) + 'rad/s'

						wind_syms_pos = 0.5, -1
						scale = 10

						print(b, l, ave_curvature)

						# plot motion of boat
						# trajectory
						axes.plot(positions[:,0], positions[:,1], marker=ma, linestyle=li, label=r'$\theta_w=$' + str(float(round(twp[0][0],2))) + info)# + bin_label)
						# axes.plot(positions[:,0], positions[:,1], color=line_colour, marker=ma, linestyle=li, label=r'$\theta_w=$' + str(float(round(twp[0][0],2))) + info)# + bin_label)	
						# final position
						axes.scatter(positions[:,0][-1], positions[:,1][-1], marker=ma, facecolors=fc, edgecolors=line_colour)	
						# wind direction
						axes.plot(0,0,'k>', markersize=10)


						# plot wind direction
						angle = pol2cart(twp[0])
						Q = axes.quiver(wind_syms_pos[0], wind_syms_pos[1], angle[0], angle[1], color=line_colour, units='x', scale=scale)
						axes.annotate(r'$\theta_w$', xy=(0.5, -1), xytext=(0.45, -1.3), size=16)
						plt.xlabel(r'x (m)' , fontsize=fontsize)
						plt.ylabel(r'y (m)', fontsize=fontsize)
						

					# rudder is straight 	
					else:
						final_x_coord = positions[:,0][-1]
						ave_vel = final_x_coord / len(twp)
						#info = ',  ' + r'$\bar v=$' + str(round(ave_vel, 2)) + ' m/s'
						s = 200 if b else 100 
						axes.scatter(twp[0][0], final_x_coord, marker=ma, facecolors=fc, edgecolors=line_colour, s=s)	
						plt.xlabel(r'$\theta_w$ (rads)' , fontsize=fontsize)
						plt.ylabel(r'$\bar v $ (m/s)', fontsize=fontsize)
						




def curvature(points):
	"""
	https://stackoverflow.com/questions/28269379/curve-curvature-in-numpy
	"""
	dx_dt = np.gradient(points[:, 0])
	dy_dt = np.gradient(points[:, 1])
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


#systematic()
# random_wind()
empirical()







