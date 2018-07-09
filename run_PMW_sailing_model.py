from PMW_sailing_model import *
from numpy import pi
import os, time, fnmatch
import random

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
	twp = [np.array([a, m]) for a, m in zip(angles, mags)]
	T = np.arange(len(twp))
	return T, twp


def cycle_wind_data():
	"""
	Systematially cycle through each wind speed and magnitude given in two lists
	Timesteps generated to match number of points
	"""
	twp = [np.array([twd, tws]) for twd in true_wind_dirs for tws in true_wind_speed]
	T = np.arange(len(twp))
	return T, twp


T, twp = cycle_wind_data()
T, twp = random_wind_data()


for r in rudder_angles:
	for s in sail_angles:
		main(rudder_angle = s, 
		     sail_angle = r,
		     auto_adjust_sail = True,
		     Time = T,
		     true_wind_polar = twp,
		     save_figs = True,
		     fig_location = save_location)



