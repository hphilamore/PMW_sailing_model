import numpy as np
from numpy import pi
import os, time, fnmatch
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splrep
from scipy.interpolate import splev
import scipy.interpolate

def plot_rudder(boatPos_pol,
			    boatAngle,
			    rudderAngle):
	"""
	Draw the rudder
	"""

	global_origin = np.array([0,0])
	boatPos_car = pol2cart(boatPos_pol)

	rudder = np.array([[- boat_l/2,            0],
		               [- boat_l/2 - rudder_l, 0]])

	rudder = Transform2D(rudder, np.array([- boat_l/2, 0]), rudderAngle)
	rudder = Transform2D(rudder, global_origin, boatAngle, boatPos_car)

	plt.plot(rudder[:,0], rudder[:,1],lw=1, color='m') 
	return rudder



def plot_sail(boatPos_pol,
		      boatAngle,
		      sailAngle):

	"""
	Draw the sail
	"""

	global_origin = np.array([0,0])
	boatPos_car = pol2cart(boatPos_pol)

	sail = np.array([[- sail_l/2, 0],
	                 [sail_l/2, 0. ]])

	sail = Transform2D(sail, global_origin, sailAngle)
	sail = Transform2D(sail, global_origin, boatAngle, boatPos_car)
	plt.plot(sail[:,0], sail[:,1],lw=1, color='k') 
	return sail



def plot_boat(boatPos_pol,
			  boatAngle,
			  sailAngle,
			  rudderAngle):
	"""
	Draw the boat
	"""

	global_origin = np.array([0,0])

	boatPos_car = pol2cart(boatPos_pol)
	# print('boat_pos', boatPos_car)


	# coords of initial poition of boat centre
	boat = np.array([[boat_l/2,  boat_w/2],
					 [-boat_l/2, boat_w/2],
					 [-boat_l/2, -boat_w/2],
					 [boat_l/2,  -boat_w/2],
					 [boat_l/2,  boat_w/2]])

	boat = Transform2D(boat, global_origin, boatAngle, boatPos_car)

	#ax1.annotate(str(i), xy=pol2cart(data["position"][i]), xytext=pol2cart(data["position"][i]) + np.array([0.2,0.2]))#, xytext=(3, 1.5),
	
	plt.scatter(boatPos_car[x], boatPos_car[y], color='k')
	plt.plot(boat[:,0], boat[:,1],lw=1, color='b') 
	
	
	
	ax1.set_aspect('equal', 'box') # equal aspect ratio, tight limits
	ax1.axis('equal')              # equal aspect ratio

	

def draw_vectors(rudder, sail, 
	             Ls_pol, Lr_pol, Lh_pol, 
	             Ds_pol, Dr_pol, Dh_pol, 
	             Fs_pol, Fr_pol, Fh_pol,
	             pos_pol, v_pol,
	             tw_pol, aw_pol,
	             surge_pol, sway_pol, 
	             Fr_moment_pol, Fh_moment_pol):

	"""
	Draw velocity and force vectors
	"""

	#print('Fr_pol2', Fr_pol)
	#print('Fr_moment_pol2', Fr_moment_pol)

	#print('surge_sway', surge_pol, sway_pol)

	Ls_car = pol2cart(Ls_pol)
	Lr_car = pol2cart(Lr_pol)
	Lh_car = pol2cart(Lh_pol)

	Ds_car = pol2cart(Ds_pol)
	Dr_car = pol2cart(Dr_pol)
	Dh_car = pol2cart(Dh_pol)

	Fs_car = pol2cart(Fs_pol)
	Fr_car = pol2cart(Fr_pol)
	Fh_car = pol2cart(Fh_pol)

	# print('vectorFs_pol', Fs_pol)
	#print('theta', theta)
	#print('vectorFs_car', Fs_car)

	pos_car = pol2cart(pos_pol)
	v_car = pol2cart(v_pol)
	aw_car = pol2cart(aw_pol)
	tw_car = pol2cart(tw_pol)

	surge_car = pol2cart(surge_pol)
	sway_car = pol2cart(sway_pol)
	Fr_moment_car = pol2cart(Fr_moment_pol)
	Fh_moment_car = pol2cart(Fh_moment_pol)



	# COEsail = np.array([
	# 	               [min(sail[:,0]) + abs(np.subtract(sail[0,0], 
	# 	               	                                 sail[1,0]))],
	# 	               [min(sail[:,1]) + abs(np.subtract(sail[0,1], 
	# 	               	                                 sail[1,1]))]
	# 	               ])


	COEsail = np.array([np.mean(sail[:,0]), 
		                  np.mean(sail[:,1])
		                  ])


	# plt.plot(COEsail[x], COEsail[y], 'go')
	# plt.plot(sail[:,0], sail[:,1], 'k*')



	# COErudder = np.array([
	# 	               [min(rudder[:,0]) + abs(np.subtract(rudder[0,0], 
	# 	               	                                   rudder[1,0]))],
	# 	               [min(rudder[:,1]) + abs(np.subtract(rudder[0,1], 
	# 	               	                                   rudder[1,1]))]
	# 	               ])

	COErudder = np.array([np.mean(rudder[:,0]), 
		                  np.mean(rudder[:,1])
		                  ])
	               # [min(rudder[:,0]) + abs(np.subtract(rudder[0,0], 
	               # 	                                   rudder[1,0]))],
	               # [min(rudder[:,1]) + abs(np.subtract(rudder[0,1], 
	               # 	                                   rudder[1,1]))]
	               # ])

	# print('COErudder', COErudder)
	# plt.plot(COErudder[x], COErudder[y], 'ro')
	# plt.plot(rudder[:,0], rudder[:,1], 'bo')



	vectors = [
                   #[COErudder[x]   , COErudder[y], Lr_car[x], Lr_car[y], 'Lrud'],
                   #[COErudder[x]   , COErudder[y], Dr_car[x], Dr_car[y], 'Drud'],
                   [pos_car[x], pos_car[y], Lh_car[x], Lh_car[y], 'Lhull'],
                   [pos_car[x], pos_car[y], Dh_car[x], Dh_car[y], 'Dhull'],
                   #[pos_car[x], pos_car[y], Ds_car[x], Ds_car[y], 'Dsail'],
                    # sail lift   
                   [pos_car[x], pos_car[y], tw_car[x],  tw_car[y], 'tw'],
                   [pos_car[x], pos_car[y], aw_car[x],  aw_car[y], 'aw'],    
                   # #[pos_car[x], pos_car[y], Ls_car[x], Ls_car[y], 'Lsail'],
                   
                   [pos_car[x],             pos_car[y], Fs_car[x],  Fs_car[y], 'Fs'],
                   [pos_car[x],             pos_car[y], Fh_car[x],  Fh_car[y], 'Fh'],
                   [COErudder[x]   , COErudder[y], Fr_car[x],  Fr_car[y], 'Fr'],
                   # #[COErudder[x]   , COErudder[y], Fr_car[x],  Fr_car[y], 'Fr'],
                   
                   [pos_car[x], pos_car[y], sway_car[x],  sway_car[y], 'Fsway'],
                   [pos_car[x], pos_car[y], surge_car[x],  surge_car[y], 'Fsurge'],
                   #[pos_car[x], pos_car[y], v_car[x],  v_car[y], 'v'],
                   #[COErudder[x]   , COErudder[y], Fr_moment_car[x],  Fr_moment_car[y], 'Fr_moment'],
                   #[pos_car[x], pos_car[y], Fh_moment_car[x],  Fh_moment_car[y], 'Fh_moment']
                   ]


	colors = cm.rainbow(np.linspace(0, 1, len(vectors)))

	#for n, (V, c, label) in enumerate(zip(vectors, colors, labels), 1):
	for n, (V, c) in enumerate(zip(vectors, colors), 1):
		# ax1.quiver(V[0], V[1], V[2], V[3], color=c, scale=5)
		quiver_scale = 2#10 # 50 #10
		Q = plt.quiver(V[0], V[1], V[2], V[3], color=c, scale=quiver_scale)
		#plt.quiverkey(Q, -1.5, n/2-2, 0.25, label, coordinates='data')
		quiver_key_scale = quiver_scale/10#100
		plt.quiverkey(Q, 1.05 , 1.1-0.1*n, quiver_key_scale, V[4], coordinates='axes')




def individual_boat_force_vector_plots(data, output_plot_title, save_figs=False):

	fig1, ax1 = plt.subplots()
	for i in data['Time']:

		#print('boat position', data["position"][i])

		# print()
		# print('saved_sail_force2', data['sail_force'][i])
		# print('heading', data['heading'][i])

		ax1.annotate(str(i), 
		             xy=pol2cart(data["position"][i]), 
		             xytext=pol2cart(data["position"][i]) + np.array([0.2,0.2]))

		# ax1.annotate(str(i), xy=data["position"][i], xytext=data["position"][i] + np.array([0.2,0.2]))#, xytext=(3, 1.5),
            #arrowprops=dict(facecolor='black', shrink=0.05),
           # )


		plot_boat(data["position"][i], 
			      data["heading"][i], 
			      data["sail_angle"][i],
			      data["rudder_angle"][i])

		rudder = plot_rudder(data["position"][i], 
			                 data["heading"][i],
			                 data["rudder_angle"][i])

		sail = plot_sail(data["position"][i], 
			             data["heading"][i], 
			             data["sail_angle"][i])

		# print()
		# print('saved_sail_force2', data['sail_force'][i])
		# print('heading', data['heading'][i])

		draw_vectors(rudder, sail, 
			         data['sail_lift'][i],   data['rudder_lift'][i],  data['hull_lift'][i],
			         data['sail_drag'][i],   data['rudder_drag'][i],  data['hull_drag'][i], 
			         data['sail_force'][i],  data['rudder_force'][i], data['hull_force'][i],
			         data['position'][i],    data["velocity"][i],
			         data["true_wind"][i],   data["apparent_wind"][i],
			         data['surge_force'][i], data['sway_force'][i],   
			         data["rudder_moment_force"][i], data["hull_moment_force"][i])

	#title = f'r_{round(ra, 3)} s_{round(sa,3)} tw_{round(tw_pol[0],3)}, {round(tw_pol[1],3)}'	
	title = output_plot_title
	plt.title(title)	

	if save_figs:
		save_fig(fig_location, title)
	else:
		plt.show()






	# else: 
	# 	ax = plot_style[2]
	# 	position = np.vstack(data['position'])
	# 	line = ':' if binary_actuator else '-'

			

	# 	plt.plot(position[:,x], position[:,y])