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

# TODO
# - plot empirical data for flat plate and find actual value of S1
# - find actual drag profile for ships rudder / ships hull

# Model of the sailing performance of a boat
# Inputs : 
# Wind angle and force (global frame)
# Sail angle (local frame)
# Sail aspect ratio as function of sail angle (to model soft robotic PMW-inspired sail)
# Rudder angle (local frame)


# Model assumptions
# All hydronamic side force is translated into lateral motion (no heeling moment)
# Hull shape is aerofoil 


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

x, y = 0, 1

# p1 = 0.03;
# p2 = 40;
# p3 = -0.6;
# p4 = 200;
# p5 = 1500;
# p6 = 0.5;
# p7 = 0.5;
# p8 = 2;
# p9 = 120;
# p10 = 400;
# p11 = 0.2;


rho_air = 1.225;
rho_water = 1000;

# boat parameters
boat_l = 1
boat_w = 0.5
rudder_l = 0.4
sail_l = 0.8


# points_x = [l/2, l/2, -l/2, -l/2, l/2]
# points_y = [-w/2, w/2, w/2, -w/2, -w/2]
# points = np.array([points_x, points_y])
# points = np.reshape(points, (5, 2))

# sail / rudder area
A_s = 0.64
A_r = 0.1
A_h = 0.5
# aspect ratio
ARs = 4
ARr = 2
ARh = 2

# Maximum normal coefficient for an infinite foil
CN1max_inf_s = 1 # empirical data from Johannsen = 0.45
CN1max_inf_r = 1.5
CN1max_inf_h = 1#0.1


# hull minimum drag coefficient
CD0_hull = 0.1

c_s= sail_l  #0.2 # chord, m
t_s = 0 # thickness
c_r= rudder_l# 0.2 # chord, m
t_r = 0 # thickness
c_h= boat_l # 0.2 # chord, m
t_h = boat_l/2 # thickness

# intial conditions
pos_car = np.array([0, 0])
pos_pol = cart2pol(pos_car)
v_car = np.array([0, 0])
v_pol = cart2pol(v_car)
theta = 0;
w = 0;
tw_pol = np.array([pi , 5])
# aw_pol = appWind(tw_pol, v_pol)
# aw_car = pol2cart(aw_pol)

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



#fig1, ax1 = plt.subplots





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
		# if part == 'sail':
			# print('calculating attack angle')
			# print('local ' + part + ' angle', np.round(part_angle,2))

		# convert angles to global frame
		part_angle += boat_angle
		# if part == 'sail':
		# 	print('global ' + part + ' angle', np.round(part_angle,2))

		# check angle still expressed in 4 quadrants
		part_angle = four_quad(part_angle)
		# if part == 'sail':
		# 	print('global ' + part + ' angle_4_quad', np.round(part_angle,2))

		# 	print('global_fluid_angle', np.round(V_pol[0],2))

	else:	

		# if part == 'sail':
		# 	print('calculating attack angle')
		# 	print('local ' + part + ' angle', np.round(part_angle,2))

		# convert angles to global frame
		part_angle += boat_angle
		# if part == 'sail':
		# 	print('global ' + part + ' angle', np.round(part_angle,2))

		# check angle still expressed in 4 quadrants
		part_angle = four_quad(part_angle)
		# if part == 'sail':
		# 	print('global ' + part + ' angle_4_quad', np.round(part_angle,2))

		# 	print('global_fluid_angle', np.round(V_pol[0],2))

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
		#print('calculating lift angle : 2nd or 4th quadrant ')
		if ((2 * pi - pa_abs >  fab  > pi*3/2 - pa_abs) or 
			(pi - pa_abs     >  fab  > pi/2   - pa_abs)):
			la = fab - pi/2
		else:
			la = fab + pi/2


    
	else:	# 1st or 3rd quadrant
		#print('calculating lift angle : 1st or 3rd quadrant ')
		if (pa_abs      <  fab   <  pi/2    + pa_abs or 
			pi + pa_abs <  fab   <  pi*3/2  + pa_abs):
			la = fab + pi/2
		else:
			la = fab - pi/2

	#print('lift_angle_boat_frame', np.round(la,2))
	#print('the lift angle= ', la+theta)

	# convert angle to global refernce frame            
	la += theta
	la = four_quad(la)
	return la 

def aero_coeffs(attack_angle, aspect_ratio, chord, thickness, 
	            CN1max_infinite, CDmin_infinite, 
	            lift_scale_factor, drag_scale_factor, overall_scale_factor):
	"""
	Computes the lift abd drag coefficient for a given angle of attack.
	Considers pre and post stall condition up to 90 degrees

	"""
	# apply drag coefficient scaling factor to make sure CD > CL for hull and rudder



	a = rad2deg(attack_angle)
	AR = aspect_ratio
	c = chord
	t = thickness


	# (Fage and Johansen): symetrical body
	A0 = 0    # the attack angle where CL = 0
	CD0 = CDmin_infinite   # minimum drag coefficient (e.g. CD at A0)

    # Convert input parameters to finite airfoil
	CN1max_inf = CN1max_infinite #0.445
	ACL1_inf = 9 #degrees
	ACD1_inf = ACL1_inf
	
	CL1max_inf = cos(deg2rad(ACL1_inf)) * CN1max_inf
	CD1max_inf = sin(deg2rad(ACL1_inf)) * CN1max_inf
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

	#else:
		# Input parameters are for finite aitfoil

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

	# apply drag scale factor so that CD > CL for rudder and hull
	#CD1 *= drag_scale_factor


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
	
	# apply drag scale factor so that CD > CL for rudder and hull
	#CD2 *= drag_scale_factor	

	#print(CL1, CL2)
	CL = max(CL1, CL2)

	#print(CD1, CD2)
	CD = max(CD1, CD2)

	# apply drag scale factor so that CD > CL for rudder and hull
	#CD *= drag_scale_factor
	CL *= (lift_scale_factor * overall_scale_factor)
	CD *= (drag_scale_factor * overall_scale_factor)

	#print('CL= ' , CL)
	#print('CD= ' , CD)

	return CL, CD, #CL1, CD1, CL2, CD2, CL1max_inf, CD1max_inf, CL1max, CD1max

#aero_coeffs = np.vectorize(aero_coeffs)

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
	    CN1max_inf = CN1max_inf_s
	    CD0 = 0
	    c = c_s
	    t = t_s
	    lift_scaler = 1
	    drag_scaler = 1
	    overall_scaler = 1
	    
	elif part == 'rudder':
	    #V_pol = v_pol
	    # v_fluid_car = v_car
	    #d = dr
	    rho = rho_water
	    AR = ARr  # aspect ratio
	    A = A_r   # area
	    CN1max_inf = CN1max_inf_r
	    CD0 = 0
	    c = c_r
	    t = t_r
	    lift_scaler = 1
	    drag_scaler = 0.02
	    overall_scaler = 0.1

	else: # part == 'hull'
		rho = rho_water
		AR = ARh  # aspect ratio
		A = A_h   # area
		CN1max_inf = CN1max_inf_h
		CD0 = CD0_hull
		c = c_h
		t = t_h
		lift_scaler = 1
		drag_scaler = 4
		overall_scaler = 1

	
	# angle of attack    
	alpha = attack_angle(part_angle, boat_angle, A, part, apparent_fluid_velocity = V_pol)

	V_pol = apparent_fluid_velocity

	# if part == 'sail':
	# 	print(part, force, 'alpha', np.round(alpha,2))

	# plot CL and CD across the whole attack angle range (0 --> pi rads)
	attack_a = np.linspace(0, pi, 1000)
	cl, cd = [], []
	for a in attack_a:
		CL, CD = aero_coeffs(attack_angle=a , 
			  				 aspect_ratio=AR, 
			  				 chord=c, 
			  				 thickness=t,
			  				 CN1max_infinite=CN1max_inf,
			  				 CDmin_infinite=CD0,
			  				 lift_scale_factor=lift_scaler, 
			  				 drag_scale_factor=drag_scaler, 
			  				 overall_scale_factor = overall_scaler
			  				 )

		cl.append(CL)
		cd.append(CD)

	attack_a= rad2deg(attack_a)
	plt.plot(attack_a, cl, label='lift')
	plt.plot(attack_a, cd, label='drag')
	plt.legend()
	plt.title(part)
	plt.xlim((0, 90))
	plt.ylim((0, 2))
	# plt.xlim((0, 90))
	#plt.ylim((0, 2))
	#plt.show()


	CL, CD = aero_coeffs(attack_angle=alpha , 
		  				 aspect_ratio=AR, 
		  				 chord=c, 
		  				 thickness=t,
		  				 CN1max_infinite=CN1max_inf,
		  				 CDmin_infinite=CD0,
		  				 lift_scale_factor=lift_scaler, 
			  			 drag_scale_factor=drag_scaler, 
			  			 overall_scale_factor = overall_scaler
		  				 )

	
	if force == 'lift':
		C = CL
		angle = four_quad(lift_angle(part_angle, area=A, apparent_fluid_velocity = V_pol))
	else:
		C = CD
		angle = four_quad(V_pol[0])
		angle = V_pol[0]

	force = 0.5 * rho * A * V_pol[1]**2 * C

	#angle = four_quad(lift_angle(part_angle, area=A, apparent_fluid_velocity = V_pol) if (force =='lift') else V_pol[0])

	# if part == 'hull':
	# 	pass
		#angle += pi 

	#angle = four_quad(angle)




	# 	lift_force = 0.5 * rho * A * V_pol[1]**2 * CL

	# # aero force
	# if force == 'lift':
	# 	lift_force = 0.5 * rho * A * V_pol[1]**2 * CL

	# 	lift_a = lift_angle(part_angle, area=A, apparent_fluid_velocity = V_pol)

	# 	# rotate direction of force by pi rads if hull force
	# 	if part == 'hull':
	# 		lift_a += pi

	# 	lift_a = four_quad(lift_a)

	# 	return np.array([lift_a, lift_force])


	# else: # force = 'drag'

	# 	drag_force = 0.5 * rho * A * V_pol[1]**2 * CD
		
	# 	drag_a = V_pol[0]

	# 	# rotate direction of force by pi rads if hull force
	# 	if part == 'hull':
	# 		drag_a += pi

	# 	drag_a = four_quad(drag_a)
		
	return np.array([angle, force])


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
	             Ls_pol, Lr_pol, Lh_pol, 
	             Ds_pol, Dr_pol, Dh_pol, 
	             Fs_pol, Fr_pol, Fh_pol,
	             pos_pol, v_pol,
	             tw_pol, aw_pol):

	Ls_car = pol2cart(Ls_pol)
	Lr_car = pol2cart(Lr_pol)
	Lh_car = pol2cart(Lh_pol)

	Ds_car = pol2cart(Ds_pol)
	Dr_car = pol2cart(Dr_pol)
	Dh_car = pol2cart(Dh_pol)

	Fs_car = pol2cart(Fs_pol)
	Fr_car = pol2cart(Fr_pol)
	Fh_car = pol2cart(Fh_pol)

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


	vectors = [
                   [pos_car[x],          pos_car[y], Ls_car[x], Ls_car[y], 'Lsail'],
                   [pos_car[x],          pos_car[y], Ds_car[x], Ds_car[y], 'Dsail'],
                   [pos_car[x]-boat_l/2, pos_car[y], Lr_car[x], Lr_car[y], 'Lrud'],
                   [pos_car[x]-boat_l/2, pos_car[y], Dr_car[x], Dr_car[y], 'Drud'],
                   # [pos_car[x], pos_car[y], v_car[x],  v_car[y], 'v'], # sail lift   
                   [pos_car[x], pos_car[y], aw_car[x],  aw_car[y], 'aw'],          
                   # [pos_car[x], pos_car[y], tw_car[x],  tw_car[y], 'tw']
                   [pos_car[x],             pos_car[y], Fs_car[x],  Fs_car[y], 'Fs'],
                   [pos_car[x]-boat_l/2   , pos_car[y], Fr_car[x],  Fs_car[y], 'Fr']
                   ]


	colors = cm.rainbow(np.linspace(0, 1, len(vectors)))

	#for n, (V, c, label) in enumerate(zip(vectors, colors, labels), 1):
	for n, (V, c) in enumerate(zip(vectors, colors), 1):
		# ax1.quiver(V[0], V[1], V[2], V[3], color=c, scale=5)
		quiver_scale = 100
		Q = plt.quiver(V[0], V[1], V[2], V[3], color=c, scale=quiver_scale)
		#plt.quiverkey(Q, -1.5, n/2-2, 0.25, label, coordinates='data')
		quiver_key_scale = quiver_scale/10#100
		plt.quiverkey(Q, 1.05 , 1.1-0.1*n, quiver_key_scale, V[4], coordinates='axes')


	# #ax1.autoscale(enable=True, axis='both', tight=None)
	# ax1.set_xlim([0, 4])
	# ax1.set_ylim([0, 4])

def dvdt(v_pol, Fs_pol, Fr_pol, Fh_pol, theta):
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

	# Sail, rudder, hull, forces in boat frame of ref
	Fs_pol[0] -= theta 
	Fr_pol[0] -= theta 
	Fh_pol[0] -= theta


	#print()

	# convert to cartesian coords
	Fs_car = pol2cart(Fs_pol)
	Fr_car = pol2cart(Fr_pol)

	Fh_car = pol2cart(Fh_pol)
	v_car = pol2cart(v_pol)

	# print('Fs_car', Fs_car)
	# print('Fr_car', Fr_car)
	# print('Fh_car', Fh_car)

	# if abs(Fh_car[x]) > abs(Fs_car[x] + Fr_car[x]):
	# 	raise ValueError("Hull drag greater then sail thrust force - redesign input parameters")

	thrust = Fs_car[x] + Fr_car[x] #+ Fh_car[x]

	# CHs = 1
	# hull_side_resistance = 0.5 * rho_water * hull_side_area * -(v_car[y]**2) * CH
	# CHf = 0.1
	# hull_side_resistance = 0.5 * rho_water * hull_side_area * -(v_car[y]**2) * CH

	# if abs(Fh_car[y]) > abs(Fs_car[y] + Fr_car[y]):
	# 	raise ValueError("Hull drag greater then sail side force - redesign input parameters")

	side_force = Fs_car[y] + Fr_car[y] #+ Fh_car[y] #+ hull_side_resistance

	F_car = np.array([thrust, side_force])
	# sum forces along each axis that result in linear travel (rudder side force assumend to reult in moment only)
	# Fcar = np.array(Fs_car[0] + Fr_car[0], 
	# 	             f_leering(Fs_car[1])) 

	# convert to polar coordinates in global frame
	Fpol = cart2pol(F_car)
	Fpol[0] += theta
	
	# convert to acceleration
	mass = 10
	
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

def dpdt(v_pol, Fs_pol, Fr_pol, Fh_pol, theta):
	"""
	The vertical velocity of the PMW. 
	"""
	#return v_pol + dvdt()

	# current vel cartesian
	v_car = pol2cart(v_pol)

	# change in vel cartesian
	dvdt_car = pol2cart(dvdt(v_pol, Fs_pol, Fr_pol, Fh_pol, theta))

	# print('dvdt_pol', cart2pol(dvdt_car))
	# print('dpdt_pol', cart2pol(v_car + dvdt_car))
	# print()
	print('dvdt_car', dvdt_car)
	print('dpdt_car', (v_car + dvdt_car))
	print()
	# print()
	# print()

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

	#print('true_wind', tw_pol)

	global vpol, pos_pol

	pos_pol = Z_state[0]
	v_pol =   Z_state[1]
	theta =   Z_state[2]
	w =       Z_state[3]

	#print('boat velocity', v_pol)

	#print('theta', theta)

	global aw_pol

	aw_pol = appWind(tw_pol, v_pol)

	vw_pol = np.array([four_quad(v_pol[0]+pi), 
		                         v_pol[1]])

	#print('aw_pol', aw_pol)
	#print()
	
	# calculate lift and drag force
	Ls_pol = aero_force(part='sail',   force='lift', apparent_fluid_velocity=aw_pol,   part_angle=sa, boat_angle=theta)
	Ds_pol = aero_force(part='sail',   force='drag', apparent_fluid_velocity=aw_pol,   part_angle=sa, boat_angle=theta)
	Lr_pol = aero_force(part='rudder', force='lift', apparent_fluid_velocity=vw_pol, part_angle=ra, boat_angle=theta)  
	Dr_pol = aero_force(part='rudder', force='drag', apparent_fluid_velocity=vw_pol, part_angle=ra, boat_angle=theta) 
	Lh_pol = aero_force(part='hull',   force='lift', apparent_fluid_velocity=vw_pol, part_angle=theta, boat_angle=theta)  
	Dh_pol = aero_force(part='hull',   force='drag', apparent_fluid_velocity=vw_pol, part_angle=theta, boat_angle=theta) 


	data["apparent_wind"].append(aw_pol)
	print('aw_pol', aw_pol)
	print('aw_car', pol2cart(aw_pol))
	print()
	# print('main_aw_car', aw_car)
	# print('main_drag_car', Ds_car)

	# print()
	print('lift sail polar', np.round(Ls_pol, 2))
	print('drag sail polar', np.round(Ds_pol, 2))
	print()
	print('lift rudder polar', np.round(Lr_pol, 2))
	print('drag rudder polar', np.round(Dr_pol, 2))
	print()
	# print('lift hull polar', np.round(Lh_pol, 2))
	# print('drag hull polar', np.round(Dh_pol, 2))
	# print()
	# print('lift sail cart', np.round(pol2cart(Ls_pol), 2))
	# print('drag sail cart', np.round(pol2cart(Ds_pol), 2))
	# print()
	# print('lift rudder cart', np.round(pol2cart(Lr_pol), 2))
	# print('drag rudder cart', np.round(pol2cart(Dr_pol), 2))
	# print()
	# print('lift hull cart', np.round(pol2cart(Lh_pol), 2))
	# print('drag hull cart', np.round(pol2cart(Dh_pol), 2))
	# print()
	#print()
	# print('lift rudder', np.round(Lr_pol, 2))
	# print('drag rudder', np.round(Dr_pol, 2))

	Fs_pol = sumAeroVectors(Ls_pol, Ds_pol)  
	Fr_pol = sumAeroVectors(Lr_pol, Dr_pol)
	Fh_pol = sumAeroVectors(Lh_pol, Dh_pol)

	print('force_sail_cart',   np.round(pol2cart(Fs_pol), 2))
	print('force_rudder_cart', np.round(pol2cart(Fr_pol), 2))
	#print('force_hull_cart',   np.round(pol2cart(Fh_pol), 2))
	print()


	
	data["sail_angle"].append(sa)
	data["rudder_angle"].append(ra)
	data["sail_area"].append(A_s)

	data['sail_lift'].append(Ls_pol)
	data['rudder_lift'].append(Lr_pol) 
	data['hull_lift'].append(Lh_pol) 

	data['sail_drag'].append(Ds_pol)
	data['rudder_drag'].append(Dr_pol)
	data['hull_drag'].append(Dh_pol)

	data['sail_force'].append(Fs_pol)
	data['rudder_force'].append(Fr_pol)
	data['hull_force'].append(Fh_pol)

	#print('sail_angle', sa)

	# print("Fs_pol", Fs_pol)
	# print("Fr_pol", Fr_pol)

	dZdt = [dpdt(v_pol, Fs_pol, Fr_pol, Fh_pol, theta), 
	        dvdt(v_pol, Fs_pol, Fr_pol, Fh_pol, theta), 
	        dthdt(w, Fs_pol, Fr_pol, theta, ra),  
	        dwdt(w, Fs_pol, Fr_pol, theta, ra),
			]

	#pos_pol = cart2pol(pol2cart(pos_pol)+pol2cart(Z_init_state[0]))



	return np.array(dZdt)
  

# main program
time = np.arange(0, 20, 1)
time = np.arange(4)

#sail_angle, rudder_angle, sail_area, position, velocity, heading, angular_vel = [], [], [], [], [], [], []
data = {'position' : [],    'apparent_wind' : [],    
        'velocity' : [],    'angular_vel' : [],  'sail_area' : [],
        
        'sail_angle' : [],  'rudder_angle' : [], 'heading' : [],        
        'sail_lift' : [],   'rudder_lift' : [],  'hull_lift' : [], 
        'sail_drag' : [],   'rudder_drag' : [],  'hull_drag' : [],
        'sail_force' : [],  'rudder_force' : [], 'hull_force' : [],
        }

data["sail_angle"].append(sa)
data["rudder_angle"].append(ra)


# data["position"].append(Z_init_state[0])
# data["velocity"].append(Z_init_state[1])
# data["heading"].append(Z_init_state[2])	
# data["angular_vel"].append(Z_init_state[3])

for t in time:
	# print()
	# print()
	# print('Z_init_start')
	# for i in range(len(Z_init_state[:2])):
	# 	#cart2pol(pol2cart(pos_pol)+pol2cart(Z_init_state[0]))
	# 	#Z_init_state[i] = Z_init_state[i] + state[i]
	# 	print(pol2cart(Z_init_state[i]))
	# print()

	data["sail_angle"].append(sa)
	data["rudder_angle"].append(ra)	
	data["position"].append(Z_init_state[0])
	data["velocity"].append(Z_init_state[1])
	data["heading"].append(Z_init_state[2])	
	data["angular_vel"].append(Z_init_state[3])

	print('velocity_pol_start', data["velocity"][t])
	print('position_pol_start', data["position"][t])
	print()
	print('velocity_cart_start', pol2cart(data["velocity"][t]))
	print('position_cart_start', pol2cart(data["position"][t]))
	#print('theta_start', data["heading"][t])

	print()


	# print('Lists') 
	# print('position_cart_array_start', data["position"])
	# print('velocity_cart_array_start', data["velocity"])
	# print('position_cart_start', pol2cart(data["position"][t]))
	# print('velocity_cart_start', pol2cart(data["velocity"][t]))
	# print()


	state = param_solve(Z_init_state)

	#params = ['pos', 'vel', 'ang', 'ang_vel']

	# Update state variables; position, velocity, angle, ang_vel
	#for i, (z, s, p) in enumerate(zip(Z_init_state, state, params)):
	for i, (z, s) in enumerate(zip(Z_init_state, state)):
		# print(p)
		# print(type(z))
		# if array (coords), convert to cartesian, add, convert back to polar
		if type(z) == np.ndarray:
			# print(pol2cart(z))
			# print(pol2cart(s))
			# print(pol2cart(z)+pol2cart(s))
			Z_init_state[i] = cart2pol(pol2cart(z)+pol2cart(s))
		# if scaler (angle), simply add
		else:
			Z_init_state[i] = z + s



	# print('Lists') 
	# print('position_cart_array_end', data["position"])
	# print('velocity_cart_array_end', data["velocity"])
	print('velocity_cart_end', pol2cart(Z_init_state[1]))
	print('position_cart_end', pol2cart(Z_init_state[0]))
	#print('theta_end', Z_init_state[2])
	print()
	print()
	print()
	# print()
	# print()
### solve using ode solver
#state = odeint(param_solve, Z_init_state, time)


# for position, heading, sail_angle, rudder_angle in zip(data["position"], 
# 													   data["heading"],	
# 													   data["sail_angle"],
# 													   data["rudder_angle"]):
fig1, ax1 = plt.subplots()




for i in time:


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
		         data['sail_lift'][i],   data['rudder_lift'][i],  data['hull_lift'][i],
		         data['sail_drag'][i],   data['rudder_drag'][i],  data['hull_drag'][i], 
		         data['sail_force'][i],  data['rudder_force'][i], data['hull_force'][i],

		         data['position'][i],    data["velocity"][i],
		         tw_pol,                 data["apparent_wind"][i])





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


# plt.ylim((-2,10))
# plt.xlim((-24,0))
plt.show()

	







