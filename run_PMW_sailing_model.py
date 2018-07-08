from PMW_sailing_model import *
from numpy import pi
import os, time, fnmatch

# main()

rudder_angles = [pi/4, pi/8, 0, -pi/8, -pi/4]
sail_angles = [0, pi/6, pi/3, pi/2, pi*2/3, pi*5/6, pi, pi+pi/6, pi+pi/3, pi+pi/2, pi+pi*2/3, pi+pi*5/6, 2*pi]

#rudder_angles = [pi/4]
# sail_angles = [pi+pi*2/3]

rudder_angles = [0]
sail_angles = [0]

sail_angles = [0, pi/6, pi/3, pi/2, pi*2/3]

rudder_angles = [pi/8]
sail_angles = [pi/6]

true_wind_dirs = [0, pi/6, pi/3, pi/2, pi*2/3, pi*5/6, pi, pi+pi/6, pi+pi/3, pi+pi/2, pi+pi*2/3, pi+pi*5/6, 2*pi]
true_wind_speed = [5]


for root, dirs, files in os.walk("/Users/hemma/Documents/Projects"):
		for d in dirs:    
			if fnmatch.fnmatch(d, "sailing_simulation_results"):
				save_location = os.path.join(root, d) + '/' + time.strftime('%Y-%m-%d--%H-%M-%S')
				os.makedirs(save_location, exist_ok=True)


for r in rudder_angles:
	for s in sail_angles:
		for twd in true_wind_dirs:
			for tws in true_wind_speed:
				main(rudder_angle = r , 
			 		  sail_angle = s,
			 		  true_wind_polar = np.array([twd, tws]),
			 		  save_figs = True,
			 		  fig_location = save_location)
