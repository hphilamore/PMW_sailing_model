import numpy as np
from numpy import pi, sin, cos, rad2deg, deg2rad, arctan2, sqrt, exp
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.lines as mlines
import matplotlib.cm as cm
import matplotlib.animation as animation
from scipy.integrate import odeint

# Model of the sailing performance of a boat
# Inputs : 
# Wind angle and force (global frame)
# Sail angle (local frame)
# Sail aspect ratio as function of sail angle (to model soft robotic PMW-inspired sail)
# Rudder angle (local frame)


# Model assumptions
# All hydronamic side force is translated into lateral motion (no heeling moment)
# Hull 

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


rho_air = 1.225;
rho_water = 1000;

# boat parameters
boat_l = 1
boat_w = 0.5
rudder_l = 0.2
sail_l = 0.8

hull_side_area = 0.001
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

c = 0.2 # chord, m
t = 0 # thickness

# intial conditions
pos_car = np.array([0, 0])
pos_pol = cart2pol(pos_car)
v_car = np.array([0, 0])
v_pol = cart2pol(v_car)
theta = 0;
w = 0;
tw_pol = np.array([pi , 5])
aw_pol = appWind(tw_pol, v_pol)
aw_car = pol2cart(aw_pol)

### SAIL AND RUDDER ANGLE IN BOAT FRAME, ALL OTHERS IN GLOBAL FRAME
# rudder angle if you are standing on boat
# % -ve towards port (left)
# % +ve towards starboard (right)
# sail angle if you are standing on boat
# angle of stern-ward end of sail to follow RH coord system
ra = 0
sa = pi/2#pi/6


#Z_init_state = [pos_car[x], pos_car[y], theta, v_pol[1] , w]
Z_init_state = [pos_pol, 
				v_pol,
				theta,
				w] 



#fig1, ax1 = plt.subplots()


def cart2pol(coords):
	x = coords[0]
	y = coords[1]
	rho = sqrt(x**2 + y**2)
	phi = arctan2(y, x)
	if phi < 0:
		phi += 2*pi
		#return(rho, phi)
	#return np.array([rho, phi])
	return np.array([phi, rho])

def pol2cart(coords):
	phi = coords[0]
	rho = coords[1]
	x = rho * cos(phi)
	y = rho * sin(phi)
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





def attack_angle(part_angle, boat_angle, area, part, apparent_fluid_velocity):
	"""
	% - inputs:
	%          A : plane area of sail or wing
	%          vfluid....................... 2x1 array
	%          v_fluid : fluid velocity, manitude and angle relative to PMW
	%          frame
	%          d : sail or rudder angle      
	% - output:
	%      
    alpha : the smallest angle between the two vectors, always positive
	"""



	# if part == 'sail': 
	# 	V_pol = aw_pol 
	# else:
	# 	V_pol = v_pol

	V_pol = apparent_fluid_velocity




	if V_pol[1] == 0: # if fluid (i.e. boat) not moving
		alpha = 0 		    # angle of attack defaults to 0
		if part == 'sail':
			print('calculating attack angle')
			print('local ' + part + ' angle', np.round(part_angle,2))

		# convert angles to global frame
		part_angle += boat_angle
		if part == 'sail':
			print('global ' + part + ' angle', np.round(part_angle,2))

		# check angle still expressed in 4 quadrants
		part_angle = four_quad(part_angle)
		if part == 'sail':
			print('global ' + part + ' angle_4_quad', np.round(part_angle,2))

			print('global_fluid_angle', np.round(V_pol[0],2))

	else:	

		if part == 'sail':
			print('calculating attack angle')
			print('local ' + part + ' angle', np.round(part_angle,2))

		# convert angles to global frame
		part_angle += boat_angle
		if part == 'sail':
			print('global ' + part + ' angle', np.round(part_angle,2))

		# check angle still expressed in 4 quadrants
		part_angle = four_quad(part_angle)
		if part == 'sail':
			print('global ' + part + ' angle_4_quad', np.round(part_angle,2))

			print('global_fluid_angle', np.round(V_pol[0],2))

		# convert angles to cartesian
		plane_car = pol2cart([part_angle, 1])
		v_fluid_car = pol2cart(V_pol)


		# use dot product to find angle cosine
		U = plane_car
		V = v_fluid_car
		cosalpha = np.dot(U, V) / np.dot(np.linalg.norm(U), np.linalg.norm(V))
	
		alpha = abs(np.arccos(cosalpha))

		# find smallest of two possible angles
		if alpha > pi/2:
			alpha = pi - alpha

	return alpha


def lift_angle(part_angle, area, apparent_fluid_velocity):
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
	
	V_pol = apparent_fluid_velocity

	# dummy cartesian coords in local boat frame of ref 
	#part_angle_car 
	dummy_len = 1
	pa_car = pol2cart([part_angle, dummy_len])
	#pa_car = pol2cart(part_angle)

	# absolute angle of sail or rudder as a refernce angle for computing 
	# 1. angle of attack  	
	# 2. direction of lift     
	#p 
	pa_abs= np.arctan(abs(pa_car[y])/ 
		     	      abs(pa_car[x]))

	
	# fluid velocity angle in local boat frame of ref
	fab = V_pol[0] - theta
	# make sure angle is in 4-quad represetation
	fab = four_quad(fab)
	# if fab > 2*pi:
	# 	fab -= 2*pi
	# elif fab < 0:
	# 	fab += 2*pi 

	#v_fluid_boat_pol = pol2cart([vfab, V_pol[1]])


	#establish orientation or sail or rudder  

	if (safe_div(pa_car[x], pa_car[y])) < 0:	# 2nd or 4th quadrant 
		print('calculating lift angle : 2nd or 4th quadrant ')
		if ((2 * pi - pa_abs >  fab  > pi*3/2 - pa_abs) or 
			(pi - pa_abs     >  fab  > pi/2   - pa_abs)):
			la = fab - pi/2
		else:
			la = fab + pi/2


    
	else:	# 1st or 3rd quadrant
		print('calculating lift angle : 1st or 3rd quadrant ')
		if (pa_abs      <  fab   <  pi/2    + pa_abs or 
			pi + pa_abs <  fab   <  pi*3/2  + pa_abs):
			la = fab + pi/2
		else:
			la = fab - pi/2

	#print('lift_angle_boat_frame', np.round(la,2))
	print('the lift angle= ', la+theta)

	# convert angle back to global refernce frame            
	return la + theta

def aero_coeffs(attack_angle, aspect_ratio, chord, thickness):
	"""
	Computes the lift abd drag coefficient for a given angle of attack.
	Considers pre and post stall condition up to 90 degrees

	"""
	a = rad2deg(attack_angle)
	AR = aspect_ratio
	c = chord
	t = thickness


	# (Fage and Johansen)
	A0 = 0
	CD0 = 0
	ACL1_inf = 9 #degrees
	ACD1_inf = ACL1_inf
	CL1max_inf = cos(deg2rad(ACL1_inf))
	CD1max_inf = sin(deg2rad(ACL1_inf))
	#print(CL1max_inf)
	#print(CD1max_inf)
	S1_inf = CL1max_inf / ACL1_inf # slope of linear segment of pre-stall lift (simplified)
	#print(S1_inf)

	# Convert from infinite plate to finite plate
	ACL1 = ACL1_inf + 18.2 * CL1max_inf * AR**(-0.9)
	ACD1 = ACL1
	CD1max = CD1max_inf  + 0.28 * CL1max_inf**2 * AR**(-0.9)		# Pre stall max drag
	S1 = S1_inf / (1 + 18.2 * S1_inf * AR**(-0.9))
	CL1max = CL1max_inf * (0.67 + 0.33 * exp(-(4.0/AR)**2))	# Pre stall max lift 

	# print(CL1max)
	# print(CD1max)



	RCL1 = S1 * (ACL1 - A0) - CL1max
	N1 = 1 + CL1max / RCL1
	M = 2.0

	# Pre-stall lift
	if a >= A0:
		CL1 = S1 * (a - A0) - RCL1 * ((a - A0)/(ACL1 - A0))**N1
	else: # a < A0
		CL1 = S1 * (a - A0) + RCL1 * ((a - A0)/(ACL1 - A0))**N1


	# Pre-stall drag
	if (2*A0 - ACD1) <= a <= ACD1:
		CD1 = CD0 + (CD1max - CD0) * ((a - A0)/(ACD1 - A0))**M
	else: # (a < (2*A0 - ACD1))  or  (a > ACD1)
		CD1 = 0


	F1 = 1.190 * (1.0-(t/c)**2)
	F2 = 0.65 + 0.35 * exp(-(9.0/AR)**2.3)
	G1 = 2.3 * exp(-(0.65 * (t/c))**0.9)
	G2 = 0.52 + 0.48 * exp(-(6.5/AR)**1.1)

	# Post stall max lift and drag
	CL2max = F1 * F2
	CD2max = G1 * G2

	RCL2 = 1.632 - CL2max
	N2 = 1 + CL2max / RCL2

	# print('RCL2= ', RCL2)
	# print('N2= ', N2)

	# print('a=', a)

	# Post stall lift
	if (0 <= a < ACL1):
		CL2 = 0
	elif ACL1 <= a <= 92.0:
		CL2 = -0.032 * (a - 92.0) - RCL2 * ((92.0 - a)/51.0)**N2
	else: # a > 92.0
		#try:
		CL2 = -0.032 * (a - 92.0) + RCL2 * ((a - 92.0)/51.0)**N2
		#print('CL2"= ', CL2)
		# except RuntimeWarning:    
		# 	print("a=",a)
		


	# Post stall drag
	if (2*A0 <= a < ACL1):
		CD2 = 0
	elif ACL1 <= a:
		ang = ((a - ACD1)/(90.0 - ACD1)) * 90 
		CD2 = CD1max + (CD2max - CD1max) * sin(deg2rad(ang))
	

	#print(CL1, CL2)
	CL = max(CL1, CL2)

	#print(CD1, CD2)
	CD = max(CD1, CD2)

	#print('CL= ' , CL)
	#print('CD= ' , CD)

	return CL, CD, #CL1, CD1, CL2, CD2, CL1max_inf, CD1max_inf, CL1max, CD1max


def aero_force(part, force, apparent_fluid_velocity, part_angle, boat_angle):
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

	V_pol = apparent_fluid_velocity

	if part == 'sail':
	    #V_pol = aw_pol
	    # v_fluid_car = aw_car
	    #d = ds
	    rho = rho_air
	    # TODO : make aspect ratio and area a function of sail angle
	    AR = ARs # aspect ratio
	    A = A_s  # area
	    
	else: # part == 'rudder'
	    #V_pol = v_pol
	    # v_fluid_car = v_car
	    #d = dr
	    rho = rho_water
	    A = A_r   # aspect ratio
	    AR = ARr  # area
	
	# angle of attack    
	alpha = attack_angle(part_angle, boat_angle, A, part, apparent_fluid_velocity = V_pol)

	V_pol = apparent_fluid_velocity

	if part == 'sail':
		print(part, force, 'alpha', np.round(alpha,2))

	CL, CD = aero_coeffs(attack_angle=alpha , 
		  				 aspect_ratio=AR, 
		  				 chord=c, 
		  				 thickness=t)

	if part=='sail':
		if force=='lift':
			print('CL', part, force, np.round(CL,2))
		else:
			print('CD', part, force, np.round(CD,2))
		print()

	# aero force
	if force == 'lift':
		lift_a = lift_angle(part_angle, area=A, apparent_fluid_velocity = V_pol)
		print(part, 'lift angle', lift_a)
		lift_force = 0.5 * rho * A * V_pol[1]**2 * CL
		return np.array([lift_a, lift_force])

	else: # force = 'drag'
		drag_a = V_pol[0];
		#print(part, 'drag angle', drag_a)
		drag_force = 0.5 * rho * A * V_pol[1]**2 * CD
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


# def plot_PMW(boatPosX, 
# 			 boatPosY, 
# 			 boatAngle,
# 			 sailAngle,
# 			 rudderAngle):
# 	#fig, ax = plt.subplots()
# 	patches = []
# 	# boat = Rectangle((starboard_stern_x, 
# 	# 				  starboard_stern_y), 
# 	#                   boat_l, 
# 	#                   boat_w, 
# 	#                   angle=theta,
# 	#                   color='c', 
# 	#                   alpha=0.5)
# 	boat = Rectangle((boatPosX - boat_l, 
# 					  boatPosY - boat_w), 
# 	                  boat_l, 
# 	                  boat_w, 
# 	                  angle=boatAngle,
# 	                  color='c', 
# 	                  alpha=0.5)

# 	rudder_x = [boatPosX - boat_l/2, 
# 	            boatPosY - boat_l/2 - rudder_l]

# 	rudder_y = [boatPosY, 
# 				boatPosY]

# 	rudder_a = [boatAngle, 
# 				boatAngle + rudderAngle]

# 	rudder_transx = []
# 	rudder_transy = []

# 	for rx, ry, a in zip(rudder_x, 
# 		                 rudder_y, 
# 		                 rudder_a):	

# 		r = np.array([[rx],[ry]])

# 		R = [[cos(a), sin(a)],
# 			 [sin(a), cos(a)]]

# 		rudder_trans = np.dot(R, r)
# 		rudder_transx.append(rudder_trans[x])
# 		rudder_transy.append(rudder_trans[y])

# 	rudder = mlines.Line2D(rudder_transx, rudder_transy, linewidth=2, color='b', alpha=0.5)


# 	sail_x = [boatPosX + sail_l/2, 
# 	          boatPosY - sail_l/2]

# 	sail_y = [boatPosX, 
# 			  boatPosY]

# 	sail_angle = [boatAngle + sailAngle, 
# 				  boatAngle + sailAngle]

# 	sail_transx = []
# 	sail_transy = []

# 	for sx, sy, a in zip(sail_x, 
# 		                 sail_y, 
# 		                 sail_angle):
# 		r = np.array([[sx],[sy]])

# 		R = [[cos(a), sin(a)],
# 			 [sin(a), cos(a)]]

# 		sail_trans = np.dot(R, r)
# 		sail_transx.append(sail_trans[x])
# 		sail_transy.append(sail_trans[y])
	

# 	sail = mlines.Line2D(sail_transx, sail_transy, linewidth=2, color='k', alpha=0.5)


# 	mast = Circle((boatPosX, boatPosY), 0.01, color='k')


# 	# fig1 = plt.figure()
# 	# ax1 = fig1.add_subplot(111, aspect='equal')

# 	# fig1, ax1 = plt.subplots()


# 	ax1.add_patch(boat)
# 	ax1.add_patch(mast)
# 	ax1.add_line(rudder)
# 	ax1.add_line(sail)

# 	origin = [0], [0] # origin point

# 	V = np.array([[1,1],[-2,2],[4,-7]])
# 	# origin_x, origin_y, length_x, length_y
# 	V = np.array([[1 , 1, 0, 0],
# 		          [-2, 2, 0, 4],
# 		          [4, -7, 5, 5]])



# 	#ax1.autoscale(enable=True, axis='both', tight=None)
# 	# ax1.set_xlim([-6, 20])
# 	# ax1.set_ylim([-6, 20])
# 	#plt.show()

def Transform2D(points, origin, angle, translation=0):
	'''
	pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian
	'''

	R = np.array([[cos(angle), sin(angle)],
                  [-sin(angle), cos(angle)]])

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

	
	
	plt.scatter(boatPos_car[x], boatPos_car[y], color='k')
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

	#print('vector_DS_car', Ds_car)


	# vectors = np.array([
	#                    [pos_car[x], pos_car[y], Ls_car[x], Ls_car[y]],
	#                    [pos_car[x], pos_car[y], Ds_car[x], Ds_car[y]],
	#                    [pos_car[x], pos_car[y], Lr_car[x], Lr_car[y]],
	#                    [pos_car[x], pos_car[y], Dr_car[x], Dr_car[y]],
	#                    # [pos_car[x], pos_car[y], v_car[x],  v_car[y]], # sail lift   
	#                    [pos_car[x], pos_car[y], aw_car[x],  aw_car[y]],          
	#                    # [pos_car[x], pos_car[y], tw_car[x],  tw_car[y]]
	#                    [pos_car[x], pos_car[y], Fs_car[x],  Fs_car[y]]
	#                    ])

	vectors = [
                   [pos_car[x], pos_car[y], Ls_car[x], Ls_car[y], 'Lsail'],
                   [pos_car[x], pos_car[y], Ds_car[x], Ds_car[y], 'Dsail'],
                   # [pos_car[x], pos_car[y], Lr_car[x], Lr_car[y], 'Lrud'],
                   # [pos_car[x], pos_car[y], Dr_car[x], Dr_car[y], 'Drud'],
                   # [pos_car[x], pos_car[y], v_car[x],  v_car[y], 'v'], # sail lift   
                   [pos_car[x], pos_car[y], aw_car[x],  aw_car[y], 'aw'],          
                   # [pos_car[x], pos_car[y], tw_car[x],  tw_car[y], 'tw']
                   [pos_car[x], pos_car[y], Fs_car[x],  Fs_car[y], 'Fs']
                   ]

	


	# labels = ['Lsail', 
	# 'Dsail', 
	# 'Lrud', 
	# 'Drud', 
	# # 'v', 
	# 'aw', 
	# # 'tw'
	# 'Fs'
	# ]

		#ax1.quiver(*origin, V[:,0], V[:,1], color=['r','b','g'], scale=21)
		# QV1 = plt.quiver(x, y, u1, v1, color='r')
		# plt.quiverkey(QV1, 1.2, 0.515, 2, 'arrow 1', coordinates='data')
		

	colors = cm.rainbow(np.linspace(0, 1, len(vectors)))

	#for n, (V, c, label) in enumerate(zip(vectors, colors, labels), 1):
	for n, (V, c) in enumerate(zip(vectors, colors), 1):
		# ax1.quiver(V[0], V[1], V[2], V[3], color=c, scale=5)
		Q = plt.quiver(V[0], V[1], V[2], V[3], color=c, scale=10)
		#plt.quiverkey(Q, -1.5, n/2-2, 0.25, label, coordinates='data')
		plt.quiverkey(Q, 1.05 , 1.1-0.1*n, 0.25, V[4], coordinates='axes')


	# #ax1.autoscale(enable=True, axis='both', tight=None)
	# ax1.set_xlim([0, 4])
	# ax1.set_ylim([0, 4])

def dvdt(v_pol, Fs_pol, Fr_pol, theta):
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
	#Fs_car, Fr_car = force_wrt_boat(Fs_pol, Fr_pol, theta)

	# Sail and drag forces in boat frame of ref
	Fs_pol[0] -= theta 
	Fr_pol[0] -= theta 

	# convert to cartesian coords
	Fs_car = pol2cart(Fs_pol)
	Fr_car = pol2cart(Fr_pol)
	v_car = pol2cart(v_pol)

	thrust = Fs_car[x] + Fr_car[x]

	CHs = 1
	hull_side_resistance = 0.5 * rho_water * hull_side_area * -(v_car[y]**2) * CH
	CHf = 0.1
	hull_side_resistance = 0.5 * rho_water * hull_side_area * -(v_car[y]**2) * CH

	side_force = Fs_car[y] + Fr_car[x] + hull_side_resistance

	F_car = np.array([thrust, side_force])
	# sum forces along each axis that result in linear travel (rudder side force assumend to reult in moment only)
	# Fcar = np.array(Fs_car[0] + Fr_car[0], 
	# 	             f_leering(Fs_car[1])) 

	# convert to polar coordinates in global frame
	Fpol = cart2pol(F_car)
	Fpol[0] += theta
	
	# convert to acceleration
	mass = 1
	
	acceleration = np.array([Fpol[0], 
		                     Fpol[1]/mass])
	return acceleration


# def f_leering(side_force):
# 	"""
# 	Function relating side force on boat to side force resulting in perpendicular acceleration of boat 
# 	"""
# 	return (0.1 * side_force)


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

	dvdt_car = pol2cart(dvdt(v_pol, Fs_pol, Fr_pol, theta))

	return cart2pol(v_car + dvdt_car) 
		            


# def force_wrt_boat(Fs_pol, Fr_pol, theta):

# 	"""
# 	Resolves the sail and rudder force perpendiular and parallel to the boat (bow to stern) axis"
# 	"""

# 	# convert to boat frame of ref
# 	Fs_pol[0] -= theta 
# 	Fr_pol[0] -= theta 

# 	# convert to cartesian coords
# 	Fs_car = pol2cart(Fs_pol)
# 	Fr_car = pol2cart(Fr_pol)

# 	return Fs_car, Fr_car

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

	# Sail and drag forces in boat frame of ref
	Fs_pol[0] -= theta 
	Fr_pol[0] -= theta 

	# convert to cartesian coords
	Fs_car = pol2cart(Fs_pol)
	Fr_car = pol2cart(Fr_pol)
	#Fs_car, Fr_car = force_wrt_boat(Fs_pol, Fr_pol, theta)

	return 0

	return (Fr_car [1] * 
		   (boat_l/2 + abs(rudder_l * cos(rudder_a))) + # moment due to force on rudder
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

	print('true_wind', tw_pol)

	global vpol, pos_pol

	pos_pol = Z_state[0]
	v_pol =   Z_state[1]
	theta =   Z_state[2]
	w =       Z_state[3]

	print('boat velocity', v_pol)

	print('theta', theta)

	global aw_pol

	aw_pol = appWind(tw_pol, v_pol)

	print('aw_pol', aw_pol)
	print()
	
	# calculate lift and drag force
	Ls_pol = aero_force(part='sail',   force='lift', apparent_fluid_velocity=aw_pol, part_angle=sa, boat_angle=theta)
	Ds_pol = aero_force(part='sail',   force='drag', apparent_fluid_velocity=aw_pol, part_angle=sa, boat_angle=theta)
	Lr_pol = aero_force(part='rudder', force='lift', apparent_fluid_velocity=v_pol, part_angle=ra, boat_angle=theta)  
	Dr_pol = aero_force(part='rudder', force='drag', apparent_fluid_velocity=v_pol, part_angle=ra, boat_angle=theta) 


	data["apparent_wind"].append(aw_pol)
	# print('main_aw_car', aw_car)
	# print('main_drag_car', Ds_car)

	print()
	print('lift sail', np.round(Ls_pol, 2))
	print('drag sail', np.round(Ds_pol, 2))
	print('lift rudder', np.round(Lr_pol, 2))
	print('drag rudder', np.round(Dr_pol, 2))

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
	        dvdt(v_pol, Fs_pol, Fr_pol, theta), 
	        dthdt(w, Fs_pol, Fr_pol, theta, ra),  
	        dwdt(w, Fs_pol, Fr_pol, theta, ra),
			]



	return np.array(dZdt)
  

# main program
time = np.arange(0, 20, 1)
time = np.arange(2)

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
ƒ
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

	







