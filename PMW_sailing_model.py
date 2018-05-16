import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.lines as mlines
import matplotlib.cm as cm
import matplotlib.animation as animation
from scipy.integrate import odeint

x, y = 0, 1

p1 = 0.03;
p2 = 40;
p3 = -0.6;
p4 = 200;
p5 = 1500;
p6 = 0.5;
p7 = 0.5;
p8 = 2;
p9 = 120;
p10 = 400;
p11 = 0.2;



#fig1, ax1 = plt.subplots()


def cart2pol(coords):
	x = coords[0]
	y = coords[1]
	rho = np.sqrt(x**2 + y**2)
	phi = np.arctan2(y, x)
	if phi < 0:
		phi += 2*pi
		#return(rho, phi)
	#return np.array([rho, phi])
	return np.array([phi, rho])

def pol2cart(coords):
	phi = coords[0]
	rho = coords[1]
	x = rho * np.cos(phi)
	y = rho * np.sin(phi)
	#return(x, y)
	np.array([x,y])
	return np.array([x, y])

def safe_div(x,y):
    if y==0: return 0
    return x/y

def four_quad(angle):
	"""
	Checks the angle is expressed in 4 quadrants
	"""
	if angle > 2*pi:
		angle -= 2*pi

	elif angle < 0:
		angle += 2*pi

	return angle 


def appWind(tw_pol, v_pol):
	"""
	% Uses polar components of true wind and cartesian components of 
	% boat velocity to compute polar coordinates of apparent wind:
	% - inputs:
	%          tw : true wind velocity
	%          v : PMW velocity 
	% - output:
	%          out....................... 2x1 array
	%          out : polar coordinates of apparent wind (if standing on PWM boat)
	%				 relative to global frame 
	"""
	v_car = pol2cart(v_pol)
	tw_car = pol2cart(tw_pol)
	#print('tw_car', tw_car)

	aw_car = np.array([(tw_car[x] - v_car[x]), 
					   (tw_car[y] - v_car[y])]) 

	aw_pol = cart2pol(aw_car)

	return aw_pol

rho_air = 1.225;
rho_water = 1000;

# boat parameters
boat_l = 1
boat_w = 0.5
rudder_l = 0.2
sail_l = 0.8
# points_x = [l/2, l/2, -l/2, -l/2, l/2]
# points_y = [-w/2, w/2, w/2, -w/2, -w/2]
# points = np.array([points_x, points_y])
# points = np.reshape(points, (5, 2))

# sail / rudder area
A_s = 0.25;
A_r = 0.05;
# aspect ratio
ARs = 5
ARr = 0.5

# intial conditions
pos_car = np.array([0, 0])
pos_pol = cart2pol(pos_car)
v_car = np.array([0, 0])
v_pol = cart2pol(v_car)
theta = 0;
w = 0;
tw_pol = np.array([pi, 1])
aw_pol = appWind(tw_pol, v_pol)
aw_car = pol2cart(aw_pol)


### SAIL AND RUDDER ANGLE IN BOAT FRAME, ALL OTHERS IN GLOBAL FRAME
# rudder angle if you are standing on boat
# % -ve towards port (left)
# % +ve towards starboard (right)
ra = 0

# sail angle if you are standing on boat
# angle of stern-ward end of sail to follow RH coord system
sa = pi/2


#Z_init_state = [pos_car[x], pos_car[y], theta, v_pol[1] , w]
Z_init_state = [pos_pol, 
				v_pol,
				theta,
				w] 







  




def attack_angle(part_angle, boat_angle, area, v_fluid_pol):
	"""
	% - inputs:
	%          A : plane area of sail or wing
	%          vfluid....................... 2x1 array
	%          v_fluid : fluid velocity, manitude and angle relative to PMW
	%          frame
	%          d : sail or rudder angle      
	% - output:
	%          alpha : the smallest angle between the two vectors, always positive
	"""

	if v_fluid_pol[1] == 0: # if fluid (i.e. boat) not moving
		alpha = 0 		    # angle of attack defaults to 0
	else:	

		# convert angles to global frame
		part_angle -= boat_angle
		print('part_angle', part_angle)

		print('fluid_angle', v_fluid_pol[0])

		# check angle still expressed in 4 quadrants
		part_angle = four_quad(part_angle)

		# convert angles to cartesian
		plane_car = pol2cart([part_angle, 1])
		v_fluid_car = pol2cart(v_fluid_pol)


		# use dot product to find angle cosine
		U = plane_car
		V = v_fluid_car
		cosalpha = np.dot(U, V) / np.dot(np.linalg.norm(U), np.linalg.norm(V))
	

		alpha = abs(np.arccos(cosalpha))

		# find smallest of two possible angles
		if alpha > pi/2:
			alpha = pi - alpha

		

	return alpha


def lift_angle(part_angle, area, v_fluid_pol):
	"""
	% Lift angle 
	% - inputs:
	%          area : plane area of sail or rudder
	%          vfluid....................... 2x1 array
	%          v_fluid_car : fluid velocity, manitude and angle relative to PMW
	%          frame
	%          d : sail or rudder angle      
	% - output:
	%          lift_angle : lift angle relative to the global frame of
	%          reference
	"""
	
	# dummy cartesian coords in boat frame of ref 
	#part_angle_car 
	pa_car = pol2cart([part_angle - theta, 1])

	# absolute angle of sail or rudder as a refernce angle for computing 
	# 1. angle of attack  	
	# 2. direction of lift     
	#p 
	pa_abs= np.arctan(abs(pa_car[y])/ 
		     	      abs(pa_car[x]))

	# fluid angle in boat frame of ref
	fab = v_fluid_pol[0] - theta
	# make sure angle is in 4-quad represetation
	fab = four_quad(fab)
	# if fab > 2*pi:
	# 	fab -= 2*pi
	# elif fab < 0:
	# 	fab += 2*pi 

	#v_fluid_boat_pol = pol2cart([vfab, v_fluid_pol[1]])


	#establish orientation or sail or rudder  

	if (safe_div(pa_car[x], pa_car[y])) < 0:	# 2nd or 4th quadrant 

	    
	    if ((2 * pi - fab > fab > 3 * pi/2 - pa_abs) or 
	    	(pi - pa_abs  > fab > pi/2 - pa_abs)):

	    	la = fab - pi/2
	    else:
	    	la = fab + pi/2

    
	else:	# 1st or 3rd quadrant
	    if (pa_abs    < fab <  fab + pi/2 or 
	    	fab + pi  < fab <  3 * pi/2  + fab):

	    	la = fab + pi/2
	    else:
	        la = fab - pi/2

	# convert angle back to global refernce frame            
	return la + theta

def aero_force(part, force, v_fluid_pol, part_angle, boat_angle):
	"""
	% Lift or drag force, square of velocity neglected due to roll of body
	% - inputs:
	%          part : sail or rudder
	%          force : lift or drag
	%          AR : sail aspect ratio     
	% - output:
	%          out....................... 2x1 array
	%          out : [angle (global ref frame), magnitude] of lift or drag force 
	"""
	if part == 'sail':
	    #v_fluid_pol = aw_pol
	    # v_fluid_car = aw_car
	    #d = ds
	    rho = rho_air
	    # TODO : make aspect ratio and area a function of sail angle
	    AR = ARs
	    A = A_s
	    
	else: # part == 'rudder'
	    #v_fluid_pol = v_pol
	    # v_fluid_car = v_car
	    #d = dr
	    rho = rho_water
	    A = A_r
	    AR = ARr
	
	# angle of attack    
	alpha = attack_angle(part_angle, boat_angle, area=A, v_fluid_pol=v_fluid_pol)

	print(part, force, 'alpha', alpha)

	if part=='sail':
		# lift coefficient
		if alpha < pi/6:
		    CL = (9 / pi) * alpha 
		else:
		    CL = -(4.5/pi) * alpha + (4.5/2)

		# drag coefficient
		CD = 0.05 + CL**2 / (pi * AR);


	else: # part == 'rudder'
	# Investigation of a semi balanced rudder,  Lübke , 2007 
		if alpha < 0.61:
		    CL = 1.21 * alpha 
		else:
		    CL = -0.77 * alpha + 1.21

		# drag coefficient
		CD = 0.05 + CL**2 / (pi * AR);
		#CD = 0.05 + CL**2 / (pi * 5);

	print('CL', part, CL)
	print('CD', part, CD)

	# aero force
	if force == 'lift':
		lift_a = lift_angle(part_angle, area=A, v_fluid_pol=v_fluid_pol)
		print(part, 'lift angle', lift_a)
		lift_force = 0.5 * rho * A * v_fluid_pol[1]**2 * CL
		return np.array([lift_a, lift_force])

	else: # force = 'drag'
		drag_a = v_fluid_pol[0];
		print(part, 'drag angle', drag_a)
		drag_force = 0.5 * rho * A * v_fluid_pol[1]**2 * CD
		return np.array([drag_a, drag_force])


def sumAeroVectors(lift_pol, drag_pol):
	"""
	% Find the resultant of perpendicular lift or drag forces as polar
	% components.
	% - inputs:
	%          lift....................... 2x1 array
	%          lift : polar components  of lift force relative to PMW frame
	%          drag....................... 2x1 array
	%          drag : polar components  of drag force relative to PMW frame
	% - output:
	%          out....................... 2x1 array
	%          out : Polar components of resultant force [mag; angle]
	"""

	lift_car = pol2cart(lift_pol)
	drag_car = pol2cart(drag_pol)

	f_car = lift_car + drag_car
	f_pol = cart2pol(f_car)

	return f_pol


def plot_PMW(boatPosX, 
			 boatPosY, 
			 boatAngle,
			 sailAngle,
			 rudderAngle):
	#fig, ax = plt.subplots()
	patches = []
	# boat = Rectangle((starboard_stern_x, 
	# 				  starboard_stern_y), 
	#                   boat_l, 
	#                   boat_w, 
	#                   angle=theta,
	#                   color='c', 
	#                   alpha=0.5)
	boat = Rectangle((boatPosX - boat_l, 
					  boatPosY - boat_w), 
	                  boat_l, 
	                  boat_w, 
	                  angle=boatAngle,
	                  color='c', 
	                  alpha=0.5)

	rudder_x = [boatPosX - boat_l/2, 
	            boatPosY - boat_l/2 - rudder_l]

	rudder_y = [boatPosY, 
				boatPosY]

	rudder_a = [boatAngle, 
				boatAngle + rudderAngle]

	rudder_transx = []
	rudder_transy = []

	for rx, ry, a in zip(rudder_x, 
		                 rudder_y, 
		                 rudder_a):	

		r = np.array([[rx],[ry]])

		R = [[np.cos(a), np.sin(a)],
			 [np.sin(a), np.cos(a)]]

		rudder_trans = np.dot(R, r)
		rudder_transx.append(rudder_trans[x])
		rudder_transy.append(rudder_trans[y])

	rudder = mlines.Line2D(rudder_transx, rudder_transy, linewidth=2, color='b', alpha=0.5)


	sail_x = [boatPosX + sail_l/2, 
	          boatPosY - sail_l/2]

	sail_y = [boatPosX, 
			  boatPosY]

	sail_angle = [boatAngle + sailAngle, 
				  boatAngle + sailAngle]

	sail_transx = []
	sail_transy = []

	for sx, sy, a in zip(sail_x, 
		                 sail_y, 
		                 sail_angle):
		r = np.array([[sx],[sy]])

		R = [[np.cos(a), np.sin(a)],
			 [np.sin(a), np.cos(a)]]

		sail_trans = np.dot(R, r)
		sail_transx.append(sail_trans[x])
		sail_transy.append(sail_trans[y])
	

	sail = mlines.Line2D(sail_transx, sail_transy, linewidth=2, color='k', alpha=0.5)


	mast = Circle((boatPosX, boatPosY), 0.01, color='k')


	# fig1 = plt.figure()
	# ax1 = fig1.add_subplot(111, aspect='equal')

	# fig1, ax1 = plt.subplots()


	ax1.add_patch(boat)
	ax1.add_patch(mast)
	ax1.add_line(rudder)
	ax1.add_line(sail)

	origin = [0], [0] # origin point

	V = np.array([[1,1],[-2,2],[4,-7]])
	# origin_x, origin_y, length_x, length_y
	V = np.array([[1 , 1, 0, 0],
		          [-2, 2, 0, 4],
		          [4, -7, 5, 5]])



	#ax1.autoscale(enable=True, axis='both', tight=None)
	# ax1.set_xlim([-6, 20])
	# ax1.set_ylim([-6, 20])
	#plt.show()

def Transform2D(points, origin, angle, translation=0):
	'''
	pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian
	'''

	R = np.array([[np.cos(angle), np.sin(angle)],
                  [-np.sin(angle), np.cos(angle)]])

	return np.dot(points-origin, R) + origin + translation

# return dot(pts-cnt,
# 		ar([[cos(ang),sin(ang)],
# 			[-sin(ang),cos(ang)]])
# 		)+cnt

#

def plot_rudder(boatPos_pol,
			  boatAngle,
			  rudderAngle):

	global_origin = np.array([0,0])
	boatPos_car = pol2cart(boatPos_pol)

	rudder = np.array([[- boat_l/2,            0],
		                   [- boat_l/2 - rudder_l, 0]])

	rudder = Transform2D(rudder, np.array([- boat_l/2, 0]), rudderAngle)
	rudder = Transform2D(rudder, global_origin, boatAngle, boatPos_car)

	
	# plt.scatter(boatPos_car[x], boatPos_car[y])
	# plt.plot(boat[:,0], boat[:,1],lw=1) 
	# plt.plot(rudder[:,0], rudder[:,1],lw=1) 
	plt.plot(rudder[:,0], rudder[:,1],lw=1, color='m') 
	return rudder
	#ax1.set_aspect('equal', 'box')

def plot_sail(boatPos_pol,
		  boatAngle,
		  sailAngle):

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
	# #fig, ax = plt.subplots()
	# patches = []

	global_origin = np.array([0,0])

	boatPos_car = pol2cart(boatPos_pol)


	# coords of initial poition of boat centre
	boat = np.array([[boat_l/2,  boat_w/2],
					 [-boat_l/2, boat_w/2],
					 [-boat_l/2, -boat_w/2],
					 [boat_l/2,  -boat_w/2],
					 [boat_l/2,  boat_w/2]])

	boat = Transform2D(boat, global_origin, boatAngle, boatPos_car)

	
	
	plt.scatter(boatPos_car[x], boatPos_car[y])
	plt.plot(boat[:,0], boat[:,1],lw=1, color='b') 
	
	
	
	ax1.set_aspect('equal', 'box') # equal aspect ratio, tight limits
	ax1.axis('equal')              # equal aspect ratio

	

def draw_vectors(sail, rudder, 
	             Ls_pol, Lr_pol, 
	             Ds_pol, Dr_pol, 
	             Fs_pol, Fr_pol,
	             pos_pol, v_pol,
	             tw_pol, aw_pol):



	Ls_car = pol2cart(Ls_pol)
	Lr_car = pol2cart(Lr_pol)
	Ds_car = pol2cart(Ds_pol)
	Dr_car = pol2cart(Dr_pol)
	Fs_car = pol2cart(Fs_pol)
	Fr_car = pol2cart(Fr_pol)
	pos_car = pol2cart(pos_pol)
	v_car = pol2cart(v_pol)
	aw_car = pol2cart(aw_pol)
	tw_car = pol2cart(tw_pol)



	COEsail = np.array([
		               [min(sail[:,0]) + abs(np.subtract(sail[0,0], 
		               	                                 sail[1,0]))],
		               [min(sail[:,1]) + abs(np.subtract(sail[0,1], 
		               	                                 sail[1,1]))]
		               ])



	COErudder = np.array([
		               [min(rudder[:,0]) + abs(np.subtract(rudder[0,0], 
		               	                                   rudder[1,0]))],
		               [min(rudder[:,1]) + abs(np.subtract(rudder[0,1], 
		               	                                   rudder[1,1]))]
		               ])

	print('vector_DS_car', Ds_car)


	vectors = np.array([
	                   [pos_car[x], pos_car[y], Ls_car[x], Ls_car[y]],
	                   [pos_car[x], pos_car[y], Ds_car[x], Ds_car[y]],
	                   [pos_car[x], pos_car[y], Lr_car[x], Lr_car[y]],
	                   [pos_car[x], pos_car[y], Dr_car[x], Dr_car[y]],
	                   # [pos_car[x], pos_car[y], v_car[x],  v_car[y]], # sail lift   
	                   [pos_car[x], pos_car[y], aw_car[x],  aw_car[y]],          
	                   # [pos_car[x], pos_car[y], tw_car[x],  tw_car[y]]
	                   [pos_car[x], pos_car[y], Fs_car[x],  Fs_car[y]]
	                   ])

	


	labels = ['Lsail', 
	'Dsail', 
	'Lrud', 
	'Drud', 
	# 'v', 
	'aw', 
	# 'tw'
	'Fs'
	]

		#ax1.quiver(*origin, V[:,0], V[:,1], color=['r','b','g'], scale=21)
		# QV1 = plt.quiver(x, y, u1, v1, color='r')
		# plt.quiverkey(QV1, 1.2, 0.515, 2, 'arrow 1', coordinates='data')
		

	colors = cm.rainbow(np.linspace(0, 1, len(vectors)))

	for n, (V, c, label) in enumerate(zip(vectors, colors, labels), 1):
		# ax1.quiver(V[0], V[1], V[2], V[3], color=c, scale=5)
		Q = plt.quiver(V[0], V[1], V[2], V[3], color=c, scale=10)
		#plt.quiverkey(Q, -1.5, n/2-2, 0.25, label, coordinates='data')
		plt.quiverkey(Q, 1.05 , 1.1-0.1*n, 0.25, label, coordinates='axes')


	# #ax1.autoscale(enable=True, axis='both', tight=None)
	# ax1.set_xlim([0, 4])
	# ax1.set_ylim([0, 4])

def dvdt(Fs_pol, Fr_pol, theta):
	"""
	% The acceleration of the PMW in the direction of the heading (theta). 
	% Component of wind force on sail acting parallel to heading direction  
	% minus the product of:
	% - component of water force on rudder acting parallel to heading direction
	% - rudder break coefficient
	% - input:
	%          v : velocity
	%          version : old model or new model
	% - output:
	%          out : acceleration
	"""

	# forces acting forwards and sideways in the boat frame of ref
	Fs_car, Fr_car = force_wrt_boat(Fs_pol, Fr_pol, theta)



	x = Fs_car[0] + Fr_car[0]
	y = f_leering(Fs_car[1]) 

	F_car = np.array([Fs_car[0] + Fr_car[0], 
		              f_leering(Fs_car[1]) ])
	# sum forces along each axis that result in linear travel (rudder side force assumend to reult in moment only)
	# Fcar = np.array(Fs_car[0] + Fr_car[0], 
	# 	             f_leering(Fs_car[1])) 


	# convert to polar coordinates in global frame
	Fpol = cart2pol(F_car)
	
	# convert to acceleration
	mass = 1
	acceleration = np.array([Fpol[0] + theta,
		                     Fpol[1]/mass])


	return acceleration


def f_leering(side_force):
	"""
	Function relating side force on boat to side force resulting in perpendicular acceleration of boat 
	"""
	return (0.1 * side_force)


# def dxdt():
# 	"""
# 	The horizontal velocity of the PMW. 
# 	"""
# 	a_car = dvdt()
# 	print("a_car", a_car)
# 	return v_car[0] + a_car * np.cos(theta)

# def dydt():
# 	"""
# 	The vertical velocity of the PMW. 
# 	"""
# 	a_car = dvdt()
# 	return  v_car[1] + a_car * np.sin(theta)

def dpdt(v_pol, Fs_pol, Fr_pol, theta):
	"""
	The vertical velocity of the PMW. 
	"""
	#return v_pol + dvdt()

	v_car = pol2cart(v_pol)




	dvdt_car = pol2cart(dvdt(Fs_pol, Fr_pol, theta))

	return cart2pol(v_car + dvdt_car) 
		            


def force_wrt_boat(Fs_pol, Fr_pol, theta):

	"""
	Resolves the sail and rudder force perpendiular and parallel to the boat (bow to stern) axis"
	"""

	# convert to boat frame of ref
	Fs_pol[0] -= theta 
	Fr_pol[0] -= theta 

	# convert to cartesian coords
	Fs_car = pol2cart(Fs_pol)
	Fr_car = pol2cart(Fr_pol)

	return Fs_car, Fr_car

	# # component of sail force in direction of boat heading
	# Fs_pol[1] = Fs_pol[1] * np.cos(Fs_pol[0])

	# # component of sail force acting perpendicular to boat heading
	# Fsb_side_pol = np.array([Fsba, 
	# 	                Fs_pol[1]*np.sin(Fsba)])

	# # component of sail force in direction of boat heading
	# Frb_fwd_pol = np.array([Frba,
	# 	                Fr_pol[1] * np.cos(Frba)])

	# # component of sail force acting perpendicular to boat heading
	# Frb_side_pol = np.array([Frba,
	# 	                Fr_pol[1] * np.sin(Frba)])

	# return Fsb_fwd_pol, Fsb_side_pol, Frb_fwd_pol, Frb_side_pol


def dwdt(w, Fs_pol, Fr_pol, theta, rudder_a):
	"""
	The angular velocity of the boat due to the rudder moment
	"""

	# force on boat, cartesian coords
	Fs_car, Fr_car = force_wrt_boat(Fs_pol, Fr_pol, theta)

	return 0

	return (Fr_car [1] * 
		   (boat_l/2 + abs(rudder_l * np.cos(rudder_a))) + # moment due to force on rudder
		   rotation_drag(w))							  # moment due to rotational drag



	# force on boat resulting in linear acceleration





	# sail force component in direction of boat heading
	

	# return (Fs_pol[1] * np.cos(theta) + 
	# 		f_leering(Fs_pol[1] * np.sin(theta)))

	# return ((Fs_car[x] * np.cos() + 	# sail force  
 #             Fr_car[x] * np.cos(Fr_car[y]))  	# rudder force										     
 #             / p10) 						   	# mass


def dthdt(w, Fs_pol, Fr_pol, theta, d_rudder):
	"""
	%--------------------------------------------------------------------------
	% The angular velocity of the PMW. 
	%          out : angular velocity
	%--------------------------------------------------------------------------
	"""
	return w + dwdt(w, Fs_pol, Fr_pol, theta, d_rudder)

def rotation_drag(w):
	"""
	drag force due to angular velocity
	"""
	return -w * 0.05


def param_solve(Z_state, time=np.arange(0, 20, 1)):

	global sa, ra
	"""
	Solves the time dependent parameters:
	- Boat velocity             (polar, global frame)
	- Boat acceleration         (polar, global frame)
	- Boat angular velocity     (polar, global frame)
	- Boat angular acceleration (polar, global frame)

	- Apparent wind angle (polar, boat frame)
	- Sail angle.         (polar, boat frame
	- Rudder angle.       (polar, boat frame)
	"""
	print()
	print()
	print('true_wind', tw_pol)

	pos_pol = Z_state[0]
	v_pol =   Z_state[1]
	theta =   Z_state[2]
	w =       Z_state[3]

	print('boat velocity', v_pol)

	print('theta', theta)

	aw_pol = appWind(tw_pol, v_pol)

	print('aw_pol', aw_pol)
	
	# calculate lift and drag force
	Ls_pol = aero_force(part='sail',   force='lift', v_fluid_pol=aw_pol, part_angle=sa, boat_angle=theta)
	Ds_pol = aero_force(part='sail',   force='drag', v_fluid_pol=aw_pol, part_angle=sa, boat_angle=theta)
	Lr_pol = aero_force(part='rudder', force='lift', v_fluid_pol=v_pol, part_angle=ra, boat_angle=theta)  
	Dr_pol = aero_force(part='rudder', force='drag', v_fluid_pol=v_pol, part_angle=ra, boat_angle=theta) 


	data["apparent_wind"].append(aw_pol)
	# print('main_aw_car', aw_car)
	# print('main_drag_car', Ds_car)

	print()
	print('lift sail', Ls_pol)
	print('drag sail', Ds_pol)
	print('lift rudder', Lr_pol)
	print('drag rudder', Dr_pol)

	Fs_pol = sumAeroVectors(Ls_pol, Ds_pol)  
	Fr_pol = sumAeroVectors(Lr_pol, Dr_pol)

	
	data["sail_angle"].append(sa)
	data["rudder_angle"].append(ra)
	data["sail_area"].append(A_s)
	data['sail_lift'].append(Ls_pol)
	data['sail_drag'].append(Ds_pol)
	data['rudder_lift'].append(Lr_pol) 
	data['rudder_drag'].append(Dr_pol)
	data['sail_force'].append(Fs_pol)
	data['rudder_force'].append(Fr_pol)

	print('sail_angle', sa)

	# print("Fs_pol", Fs_pol)
	# print("Fr_pol", Fr_pol)

	dZdt = [dpdt(v_pol, Fs_pol, Fr_pol, theta), 
	        dvdt(Fs_pol, Fr_pol, theta), 
	        dthdt(w, Fs_pol, Fr_pol, theta, ra),  
	        dwdt(w, Fs_pol, Fr_pol, theta, ra),
			]



	return np.array(dZdt)
  

# main program
time = np.arange(0, 20, 1)
time = np.arange(3)

#sail_angle, rudder_angle, sail_area, position, velocity, heading, angular_vel = [], [], [], [], [], [], []
data = {'apparent_wind' : [], 'sail_angle' : [], 'rudder_angle' : [], 'sail_area' : [], 
        'position' : [],   'velocity' : [],     'heading' : [], 'angular_vel' : [],
        'sail_lift' : [], 'sail_drag' : [], 'rudder_lift' : [], 'rudder_drag' : [],
        'sail_force' : [], 'rudder_force' : [],
        }

data["sail_angle"].append(sa)
data["rudder_angle"].append(ra)


# data["position"].append(Z_init_state[0])
# data["velocity"].append(Z_init_state[1])
# data["heading"].append(Z_init_state[2])	
# data["angular_vel"].append(Z_init_state[3])


for t in time:
	data["sail_angle"].append(sa)
	data["rudder_angle"].append(ra)	
	data["position"].append(Z_init_state[0])
	data["velocity"].append(Z_init_state[1])
	data["heading"].append(Z_init_state[2])	
	data["angular_vel"].append(Z_init_state[3])

	state = param_solve(Z_init_state)

	for i in range(len(Z_init_state)):
		Z_init_state[i] = Z_init_state[i] + state[i]

### solve using ode solver
#state = odeint(param_solve, Z_init_state, time)


# for position, heading, sail_angle, rudder_angle in zip(data["position"], 
# 													   data["heading"],	
# 													   data["sail_angle"],
# 													   data["rudder_angle"]):
fig1, ax1 = plt.subplots()




for i in range(len(time)):


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

	draw_vectors(rudder, sail,
		         data['sail_lift'][i],   data['rudder_lift'][i],
		         data['sail_drag'][i], data['rudder_drag'][i],
		         data['sail_force'][i],  data['rudder_force'][i],
		         data['position'][i],    data["velocity"][i],
		         tw_pol, data["apparent_wind"][i])





# for s0, s1 in zip(state[:, 0], state[:, 1]):
# 	print((s0, s1), (s1, s0))
# 	rudder = mlines.Line2D((s0, s1), (s1, s0), linewidth=2, color='b', alpha=0.5)
# 	ax2.add_line(rudder)
# plt.plot(time, state[:, 0], alpha=0.5)
# plt.plot(time, state[:, 1], alpha=0.5)
# plt.plot(time, state[:, 2], alpha=0.5)


#plot(state[:, 0], state[:, 1], 'b-', alpha=0.2)

# def animate(i):
#     plt.plot(time, state[:, 0], 'b-')
#     plt.plot(state[0:i, 0], state[0:i, 1], 'b-')

# ani = animation.FuncAnimation(fig2, animate, frames = 100, interval=200)
# ani
#plt.axes().set_aspect('equal', 'datalim')


# plt.ylim((-2.5,2.5))
# plt.xlim((-2.5,2.5))
plt.show()

	







