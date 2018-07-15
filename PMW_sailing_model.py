
"""
Sailing Model for Portugese Man of War Robot.


If running from commmand line:
- wind direction [0]
- sail angle [1]
- rudder angle [2] 
can be set as parameters. 


Inputs:
- Wind angle and force (global frame)
- Sail angle (local frame)
- Sail aspect ratio as function of sail angle (to model soft robotic PMW-inspired sail)
- Rudder angle (local frame)

Assumptions:
- All hydronamic side force is translated into lateral motion (no heeling moment)
- Hull, sail and rudder shape behaves as aerofoil 
- maximum normal force coefficent chosen to achieve characteristic post-stall lift and drag curves (empirical value 0.45)
- roll neglected for now
- water velocity in glocbal frame is neglected

TODO
MOST IMPORTANTLY
- introduce roll to deal with lateral force that curently accounted for using an arbitrary scale factor
- get rid of scaling factor for lateral force by considering smaller time increments (i.e. using ODE solver with wind as forcing function)


- plot sail area, chord, thicknes and max nornmal force coefficent should chnage dynamically with sail angle --> examine resulting drag coefficient
- use odeint solver with empirical wind data as forcing function
- only solve acceleration at each timestep - position can be derived
- find slope of linear segment of pre-stall lift (simplified) more accuratelty
- characterise real airfoil and subs in drag coefficent curve for estmaited curve in this model
- replace lift angle gernatin function with simlper lift angle function that currently isn't working for unknown reason - needs checking
- hull drag too large if following same model as used for flat pat saila nd rudder, 3 scaling factors  applied to correct drag
- scale factor also used for boat lateral force

"""

import numpy as np
from numpy import pi, sin, cos, rad2deg, deg2rad, arctan2, sqrt, exp
import matplotlib
import matplotlib.animation as animation
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from scipy.integrate import odeint
import sys, os, time, fnmatch


for root, dirs, files in os.walk("/Users/hemma/Documents/Projects"):
		for d in dirs:    
			if fnmatch.fnmatch(d, "sailing_simulation_results"):
				save_location = os.path.join(root, d) + '/' + time.strftime('%Y-%m-%d--%H-%M-%S')
				os.makedirs(save_location, exist_ok=True)


# x, y = 0, 1


def cart2pol(coords):
	"""
	Converts cartesian to polar coordinates using 4 quadrant, positive angles 0 --> 2*pi
	Returns phi, rho
	"""
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
	"""
	Converts polar to cartesian coordinates 
	Returns x, y
	"""
	phi = coords[0]
	rho = coords[1]
	x = rho * cos(phi)
	y = rho * sin(phi)
	#return(x, y)
	np.array([x,y])
	return np.array([x, y])


def safe_div(x,y):
	"""
	Divides first input by second input, returns 0 if denominator = 0
	"""
	if y==0: return 0
	return x/y


# rho_air = 1.225;   # density air
# rho_water = 1000;  # density water

# # boat mass
# mass = 10#10 # kg

# # boat geometry
# boat_l = 1         # boat length
# boat_w = 0.5       # boat width
# rudder_l = 0.4     # rudder length
# sail_l = 0.8       # sail length

# # sail / rudder area
# A_s = 0.64         # sail area
# A_r = 0.05          # rudder area
# A_h = 0.1          # hull area 

# # aspect ratio
# ARs = 4            # aspect ratio sail
# ARr = 0.02            # aspect ratio rudder
# ARh = 4            # aspect ratio hull

# # chord
# c_s= sail_l        # sail chord, m
# c_r= rudder_l      # rudder chord, m
# c_h= boat_l        # hull chord, m

# # thickness
# t_s = 0.15 * sail_l      # sail thickness
# t_r = 0.15 * rudder_l      # rudder thickness
# t_h = 0.5 * boat_l      # hull thickness

# # Maximum normal coefficient for an infinite foil
# CN1inf_s_max = 1 # empirical data from Fage and Johannsen, 1876 = 0.45
# CN1inf_r_max = 0.1
# CN1inf_h_max = 0.1

# # Minimum  normal coefficient (i.e. at angle of attack = A0) for an infinite foil
# CN1inf_s_min = 0 # empirical data from Johannsen = 0
# CN1inf_r_min = 0
# CN1inf_h_min = 0

# # Minimum drag coefficient (i.e. CD at angle of attack = A0) for an infinite foil
# CD0_r = 0
# CD0_s = 0
# CD0_h = 0.001

# # intial conditions
# pos_car = np.array([0, 0])      # boat position, GRF (global reference frame)
# pos_pol = cart2pol(pos_car)
# v_car = np.array([0, 0])        # boat velocity (x, y), GRF
# v_pol = cart2pol(v_car)
# theta = 0;                      # boat angle, GRF
# w = 0;                          # boat angular velocity, GRF
# tw_pol = np.array([pi + (pi/6), 5])   # true wind velocity (angle, magnitude), GRF
# tw_pol = np.array([2*pi - deg2rad(45) , 5])
# tw_pol = np.array([pi - deg2rad(25) , 5])
# tw_pol = np.array([2*pi , 5])

# ra = pi/10                          # rudder angle, LRF (local reference frame)
# sa = pi/2                       # sail angle, LRF

# hull_drag_scale_factor = 0.01
# hull_pre_stall_drag_scale_factor = 0.1
# hull_pre_stall_scale_factor = 8


# # initial values of time-varying parameters
# Z_init_state = [#pos_pol, 
# 				v_pol,
# 				#theta,
# 				w] 





def four_quad(angle):
	"""
	Converts angle to 4 quadrants, positive angles 0 --> 2*pi
	"""
	if angle > 2*pi:
		angle -= 2*pi

	elif angle < 0:
		angle += 2*pi

	return angle 


def appWind(tw_pol, v_pol):
	"""
	Computes polar coordinates of apparent wind:

	Inputs:
	- tw : true wind velocity
	- v  : PMW velocity 

	Output:
	- out....................... 2x1 array
	- out : polar coordinates of apparent wind (i.e. if standing on PWM boat), expresed in GRF 
	"""
	v_car = pol2cart(v_pol)
	tw_car = pol2cart(tw_pol)
	#print('tw_car', tw_car)

	aw_car = np.array([(tw_car[x] - v_car[x]), 
					   (tw_car[y] - v_car[y])]) 

	aw_pol = cart2pol(aw_car)

	return aw_pol





def attack_angle(part_angle, boat_angle, incident_vector_polar, rudder=False):
	"""
	Finds the angle of attack between:
	- a surface (e.g. sail, hull, rudder)
	- a fluid (e.g. water, wind)

    Returns the smallest of the two angles between the two vectors
    Returned value always positive
	"""
	V_pol = incident_vector_polar

	if V_pol[1] == 0:       # if fluid (i.e. boat) not moving
		alpha = 0 		    # angle of attack defaults to 0

	else:		
		if rudder:
			print('part_angle', part_angle)
			print('boat_angle', boat_angle)

		# convert angles to global frame
		part_angle += boat_angle
		if rudder:
			print('global part angle', part_angle)

		# check angle still expressed in 4 quadrants
		part_angle = four_quad(part_angle)
		if rudder:
			print('four_quad part angle', part_angle)

		# convert angles to cartesian
		part_car = pol2cart([part_angle, 1])
		#v_fluid_car = pol2cart(V_pol)
		v_fluid_car = pol2cart(np.array([V_pol[0], 1]))
		if rudder:
			print('part car', part_car)
			print('fluid car', v_fluid_car)


		# use dot product to find angle cosine
		U = part_car
		V = v_fluid_car
		cosalpha = np.dot(U, V) / np.dot(np.linalg.norm(U), np.linalg.norm(V))

		# round cosalpha to 15dp to deal with floating point error
		cosalpha = round(cosalpha, 15)
		if rudder:
			print('cosalpha', cosalpha)

		# print('part', U)
		# print('fluid', V)
		# print('cosalpha', cosalpha)
		alpha = abs(np.arccos(cosalpha))
		if rudder:
			print('alpha', alpha)

		# find smallest of two possible angles
		if alpha > pi/2:
			alpha = pi - alpha

		# print('part_angle', part_angle)
		# print('boat_angle', boat_angle)
		# print('incident_vector_polar', incident_vector_polar)
		# print('alpha', alpha)



	return alpha


def force_angle_LRF(part_angle, incident_vector_polar, force):
	"""
	Returns the angle of the lift force on a component, expressed in LRF (i.e. relative to boat)
	"""
	
	dummy_len = 1.   # dummy cartesian coords in local boat frame of ref 
	# Find minimum angle of part relative to boat (i.e. LRF)
	pa_car = pol2cart([part_angle, dummy_len])
	# Find minimum ABSOLLUTE angle of part relative to boat (i.e. LRF)
	pa_abs= np.arctan(abs(pa_car[y])/ 
		     	      abs(pa_car[x]))

	
	# Incident vector angle angle, LRF
	V_pol = incident_vector_polar 
	#print('theta3' , theta)
	fab = V_pol[0] - theta
	fab = four_quad(fab)


	# Establish orientation or sail or rudder  
	if (safe_div(pa_car[x], pa_car[y])) < 0:	# 2nd or 4th quadrant 
		#print('calculating lift angle : 2nd or 4th quadrant ')
		if ((2 * pi - pa_abs >  fab  > pi*3/2 - pa_abs) or 
			(pi - pa_abs     >  fab  > pi/2   - pa_abs)):
			la = - pi/2
		else:
			la = + pi/2

	else:	# 1st or 3rd quadrant
		#print('calculating lift angle : 1st or 3rd quadrant ')
		if (pa_abs      <  fab   <  pi/2    + pa_abs or 
			pi + pa_abs <  fab   <  pi*3/2  + pa_abs):
			la = + pi/2
		else:
			la = - pi/2

	
	# force angle is orthoganol to FLUID VELOCITY if force == lift
	# otherwise leave as orthoganol to BOAT
	if force == 'lift':
		la += fab

		
	# convert angle to global refernce frame            
	# la += theta
	# la = four_quad(la)
	return la

def lift_angle(part_angle, incident_vector_polar, boat_angle):
	"""
	Returns the angle of the lift force on a component, expressed in GRF (i.e. relative to boat)
	"""

	# First find angle of incident vector in part frame of ref
	# incident cevotr in globa frame of ref
	V_pol = incident_vector_polar
	# incident vector in boat LRF
	V_pol[0] -= boat_angle
	# incident vector in part RF
	V_pol[0] -= part_angle

	if ((0 <= V_pol[0] < pi/2) or (pi <= V_pol[0] < pi*3/2)):   # first or third quadrant relative to part
		angle = V_pol[0] + pi/2
	else: # second or fourth quadrantt relative to part
		angle = V_pol[0] - pi/2

	# convert angle back to GRF
	angle += (part_angle)# + boat_angle)
	return angle

def moment_force_angle(part_angle, incident_vector_polar, boat_angle):
	"""
	Returns the angle of the force used in a couple about the boat COG, angle is orthoganol to boat long (bow to stern) axis
	Expressed in LRF
	"""

	# First find angle of incident vector in part frame of ref
	# incident cevotr in globa frame of ref
	V_pol = incident_vector_polar

	# incident vector in boat LRF
	V_pol[0] -= boat_angle

	if not np.allclose(four_quad(part_angle), four_quad(boat_angle)):
		print('not hull')
		# incident vector in part RF
		V_pol[0] -= part_angle
	
	if (0 <= V_pol[0] < pi):   # first or second quadrant relative to part
		angle = pi/2
	else: 	                   # third or fourth quadrantt relative to part
		angle = - pi/2

	return angle

# def force_angle_LRF_(part_angle, incident_vector_polar, force):
# 	"""
# 	Returns the angle of the lift force on a component, expressed in LRF (i.e. relative to boat)
# 	"""
	
# 	dummy_len = 1.   # dummy cartesian coords in local boat frame of ref 
# 	# Find minimum angle of part relative to boat (i.e. LRF)
# 	pa_car = pol2cart([part_angle, dummy_len])
# 	# Find minimum ABSOLLUTE angle of part relative to boat (i.e. LRF)
# 	pa_abs= np.arctan(abs(pa_car[y])/ 
# 		     	      abs(pa_car[x]))

	
# 	# Incident vector angle angle, LRF
# 	V_pol = incident_vector_polar 
# 	#print('theta3' , theta)
# 	fab = V_pol[0] - theta
# 	fab = four_quad(fab)


# 	# Establish orientation or sail or rudder  
# 	if (safe_div(pa_car[x], pa_car[y])) < 0:	# 2nd or 4th quadrant 
# 		#print('calculating lift angle : 2nd or 4th quadrant ')
# 		if ((2 * pi - pa_abs >  fab  > pi*3/2 - pa_abs) or 
# 			(pi - pa_abs     >  fab  > pi/2   - pa_abs)):
# 			la = - pi/2
# 		else:
# 			la = + pi/2

# 	else:	# 1st or 3rd quadrant
# 		#print('calculating lift angle : 1st or 3rd quadrant ')
# 		if (pa_abs      <  fab   <  pi/2    + pa_abs or 
# 			pi + pa_abs <  fab   <  pi*3/2  + pa_abs):
# 			la = + pi/2
# 		else:
# 			la = - pi/2

	
# 	# force angle is orthoganol to FLUID VELOCITY if force == lift
# 	# otherwise leave as orthoganol to BOAT
# 	if force == 'lift':
# 		la += fab

		
# 	# convert angle to global refernce frame            
# 	# la += theta
# 	# la = four_quad(la)
# 	return la 

def aero_coeffs(attack_angle, AR, c, t, CN1inf_max, ACL1_inf, CD0, part):  
	"""
	Computes the lift and drag coefficient for a given angle of attack.
	Considers pre and post stall condition up to angle of attack = 90 degrees to incident surface
    Apply drag coefficient scaling factor to make sure CD > CL for hull and rudder
    Uses model from "Models of Lift and Drag Coefficients of Stalled and Unstalled Airfoils in Wind Turbines and Wind Tunnels" by David A. Spera
	"""
	# David A. Spera model uses degrees
	#print('attack_angle', attack_angle)
	a = rad2deg(attack_angle)    # angle of attack

	############################
	# David A. Spera Model 
	############################
	# Parameters for infinite foil
	# empirical data (Fage and Johansen, 1876)
	# α = angle of attack
	A0 = 0               # α, where CL = 0, 
	#ACL1_inf = 9         # α at max pre-stall lift (deg), empirical data (Fage and Johansen, 1876)
	ACD1_inf = ACL1_inf  # α at max pre-stall drag (deg)
	CL1inf_max = cos(deg2rad(ACL1_inf)) * CN1inf_max      # maximum pre-stall lift coefficient; (i.e at α = ACL1)
	CD1inf_max = sin(deg2rad(ACL1_inf)) * CN1inf_max       # maximum pre-stall drag coefficient; (i.e at α = ACD1)
	S1_inf = CL1inf_max / ACL1_inf                         # slope of linear segment of pre-stall lift (simplified : slope of line from min to max CL)


	# Convert parameters for infinite foil to finite foil	
	ACL1 = ACL1_inf + 18.2 * CL1inf_max * AR**(-0.9)         # α at max pre-stall lift (deg)
	ACD1 = ACL1 
	# if part == 'sail':                                            # α at max pre-stall drag (deg)
	# 	print('ACL1', ACL1)

	CD1max = CD1inf_max  + 0.28 * CL1inf_max**2 * AR**(-0.9) # Pre stall max drag
	CL1max = CL1inf_max * (0.67 + 0.33 * exp(-(4.0/AR)**2))	 # Pre stall max lift 
	S1 = S1_inf / (1 + 18.2 * S1_inf * AR**(-0.9))           # slope of linear segment of pre-stall lift (remember this is simplified so not accurate!)

	# Pre-stall
	RCL1 = S1 * (ACL1 - A0) - CL1max     # reduction from extension of linear segment of lift curve to CL1max
	N1 = 1 + CL1max / RCL1               # exponent defining shape of lift curve at ACL1max
	M = 2.0                              # order of pre-stall drag curve (2 --> quadratic)


	# Pre-stall lift
	if a >= A0:
		CL1 = S1 * (a - A0) - RCL1 * ((a - A0)/(ACL1 - A0))**N1
	else: # a < A0
		CL1 = S1 * (a - A0) + RCL1 * ((a - A0)/(ACL1 - A0))**N1   # inverts solution


	# Pre-stall drag
	if (2*A0 - ACD1) <= a <= ACD1:
		CD1 = CD0 + (CD1max - CD0) * ((a - A0)/(ACD1 - A0))**M
	else: # (a < (2*A0 - ACD1))  or  (a > ACD1)
		CD1 = 0

	# tc = 0.15 # 0.8 
	# asp = 4 # 4

	# F1 = 1.190 * (1.0-(tc)**2)
	# F2 = 0.65 + 0.35 * exp(-(9.0/asp)**2.3)
	# G1 = 2.3 * exp(-(0.65 * (tc))**0.9)
	# G2 = 0.52 + 0.48 * exp(-(6.5/asp)**1.1)

	F1 = 1.190 * (1.0-(t/c)**2)
	F2 = 0.65 + 0.35 * exp(-(9.0/AR)**2.3)
	G1 = 2.3 * exp(-(0.65 * (t/c))**0.9)
	G2 = 0.52 + 0.48 * exp(-(6.5/AR)**1.1)

	# Post stall max lift and drag
	CL2max = F1 * F2
	CD2max = G1 * G2

    # Post stall
	RCL2 = 1.632 - CL2max      # reduction from extension of linear segment of lift curve to CL2max exponent
	N2 = 1 + CL2max / RCL2     # exponent defining shape of lift curve at CL2max 

	# Post stall lift
	if (0 <= a < ACL1):
		CL2 = 0
	elif ACL1 <= a <= 90.0:
		CL2 = -0.032 * (a - 90.0) - RCL2 * ((90.0 - a)/51.0)**N2
	else: # a > 90.0
		CL2 = -0.032 * (a - 90.0) + RCL2 * ((a - 90.0)/51.0)**N2

	# Post stall drag
	#print('a', a)
	#print('ACL1', ACL1)
	#print('ACL1', ACL1)
	#print('a', a)
	if (2*A0 <= a < ACL1):
		CD2 = 0
	elif ACL1 <= a:
		ang = ((a - ACD1)/(90.0 - ACD1)) * 90 
		CD2 = CD1max + (CD2max - CD1max) * sin(deg2rad(ang))
		#print('CD2', CD2)


	hull_drag_scale_factor = 0.01# 0.5#0.1# 0.01
	hull_pre_stall_drag_scale_factor = 0.1
	hull_pre_stall_scale_factor = 8

	if part == 'hull': #or part == 'rudder':
		CL2 *= hull_drag_scale_factor #* 1.2
		CD2 *= hull_drag_scale_factor #* 1.2	

		CL1 *= hull_drag_scale_factor * hull_pre_stall_scale_factor
		CD1 *= hull_drag_scale_factor * hull_pre_stall_scale_factor * hull_pre_stall_drag_scale_factor

	if part == 'rudder':
		# CL2 *= rudder_scale_factor * 0.05#* 0.05
		# CD2 *= rudder_scale_factor #* 1.2	

		# CL1 *= rudder_scale_factor * 2 * 0.05#* 0.05
		# CD1 *= rudder_scale_factor * 2 * 0.7

		CL2 *= hull_drag_scale_factor * rudder_scale_factor #* 0.05#* 0.05
		CD2 *= hull_drag_scale_factor * rudder_scale_factor * 1.2	

		CL1 *= hull_drag_scale_factor * hull_pre_stall_scale_factor * rudder_scale_factor * 2 #* 0.05#* 0.05
		CD1 *= hull_drag_scale_factor * hull_pre_stall_scale_factor * hull_pre_stall_drag_scale_factor * rudder_scale_factor * 2 #* 0.7



	if part == 'sail':
		CD2 *= sail_drag_scale_factor #* 1.2
		CD1 *= sail_drag_scale_factor #* 1.2


	CL = max(CL1, CL2)

	#print(CD1, CD2)
	CD = max(CD1, CD2)

	return CL, CD


def aero_force(part, 
	           force, 
	           apparent_fluid_velocity, 
	           part_angle, 
	           boat_angle, 
	           plot_coefficients):
	"""
	Returns lift or drag force as polar coordinates in GRF
	Square of velocity neglected due to roll of body

	"""

	V_pol = apparent_fluid_velocity

	if part == 'sail':
	    rho = rho_air
	    AR = ARs # aspect ratio
	    A = A_s  # area
	    CN1inf_max = CN1inf_s_max
	    ACL1_inf = ACL1_inf_s
	    CD0 = CD0_s
	    c = c_s
	    t = t_s
	    # lift_scaler = 2
	    # drag_scaler = 0.1
	    # overall_scaler = 1
	    
	elif part == 'rudder':
	    rho = rho_water
	    AR = ARr  # aspect ratio
	    A = A_r   # area
	    CN1inf_max = CN1inf_r_max
	    ACL1_inf = ACL1_inf_r
	    CD0 = CD0_r
	    c = c_r
	    t = t_r
	    # lift_scaler = 0.1
	    # drag_scaler = 4
	    # overall_scaler = 0.1

	else: # part == 'hull'
		rho = rho_water
		AR = ARh  # aspect ratio
		A = A_h   # area
		CN1inf_max = CN1inf_h_max
		ACL1_inf = ACL1_inf_h
		CD0 = CD0_h
		c = c_h
		t = t_h
		# lift_scaler = 1
		# drag_scaler = 4
		# overall_scaler = 1

	
	# angle of attack    
	alpha = attack_angle(part_angle, boat_angle, incident_vector_polar = V_pol)

	#V_pol = apparent_fluid_velocity


	# plot CL and CD across the whole attack angle range (0 --> pi rads)
	if plot_coefficients:
		attack_a = np.linspace(0, pi, 1000)
		cl, cd = [], []
		for a in attack_a:
			#print("finding attack 1")
			CL, CD = aero_coeffs(a, AR, c, t, CN1inf_max, ACL1_inf, CD0, part)  				 
			cl.append(CL)
			cd.append(CD)
		#fig = plt.subplots()
		#if part == 'hull':
		#if part == 'rudder':
		if part == 'hull':
			attack_a= rad2deg(attack_a)
			plt.plot(attack_a, cl, label='lift '+ part)
			plt.plot(attack_a, cd, label='drag '+ part)
			#plt.legend()
			plt.title(part)
			plt.xlim((0, 90))
			plt.ylim(ymin=0)


	# find lift and drag coefficent 
	#print("finding attack 2")
	CL, CD = aero_coeffs(alpha, AR, c, t, CN1inf_max, ACL1_inf, CD0, part)

	if force == 'lift':
		C = CL
		angle = four_quad(force_angle_LRF(part_angle, V_pol, force))

		#angle = four_quad(lift_angle(part_angle, V_pol, boat_angle))

		# convert angle to global refernce frame            
		angle += theta
		angle = four_quad(angle)
	else: # force == 'drag'
		C = CD
		angle = four_quad(V_pol[0])
		#angle = V_pol[0]

	


	Force = 0.5 * rho * A * V_pol[1]**2 * C	

	# if part =='rudder':
	# 	if force == 'lift':
	# 		print('alpha=', np.rad2deg(alpha), ' CL=', C, ' lift=', Force)
	# 	else:
	# 		print('alpha=', np.rad2deg(alpha), ' CD=', C, ' drag=', Force)





	return np.array([angle, Force])


def sumAeroVectors(lift_pol, drag_pol, part):
	"""
	Returns resultant of perpendicular lift or drag forces as polar coordinates in GRF
	"""

	lift_car = pol2cart(lift_pol)
	drag_car = pol2cart(drag_pol)
	

	f_car = lift_car + drag_car

	f_pol = cart2pol(f_car)
	f_pol_c = np.array([four_quad(f_pol[0]), f_pol[1]])
	#if part == 'rudder':
		# print("lift,drag", lift_car, drag_car)
		# print("lift,drag", f_car)
		# print("lift,drag", f_pol)
		# print("lift,drag", f_pol_c)


	return f_pol

def Transform2D(points, origin, angle, translation=0):
	'''
	pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian
	'''

	R = np.array([[cos(angle), sin(angle)],
                  [-sin(angle), cos(angle)]])

	return np.dot(points-origin, R) + origin + translation




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


	# #ax1.autoscale(enable=True, axis='both', tight=None)
	# ax1.set_xlim([0, 4])
	# ax1.set_ylim([0, 4])

def dvdt(Fs_pol, Fr_pol, Fh_pol, boat_angle):

	"""
	Acceleration of the PMW, angle and magnitude, LRF

	"""

	boat_angle=four_quad(boat_angle)
	print(boat_angle)
	boat_angle=np.round(boat_angle, 8)
	print(boat_angle)
	# initial vel cartesian
	#print('vel_pol_GRF_init_', v_pol)
	#v_car = pol2cart(v_pol)
	#print('vel_car_GRF_init_', v_car)



	# CONVERT EVERYTHING TO LOCAL FRAME OF REF
	Fs_pol_LRF = np.array([Fs_pol[0]-boat_angle, Fs_pol[1]])
	Fr_pol_LRF = np.array([Fr_pol[0]-boat_angle, Fr_pol[1]])
	Fh_pol_LRF = np.array([Fh_pol[0]-boat_angle, Fh_pol[1]])


	Fs_car_LRF = pol2cart(Fs_pol_LRF)
	Fr_car_LRF = pol2cart(Fr_pol_LRF)
	Fh_car_LRF = pol2cart(Fh_pol_LRF)

	print('Fs_car', Fs_car_LRF)
	print('Fr_car', Fr_car_LRF)
	print('Fh_car', Fh_car_LRF)
	print('F_car', Fs_car_LRF + Fr_car_LRF + Fh_car_LRF)
	#v_car = pol2cart(v_pol)

	# print('local forces:')
	# print('sail', Fs_car)
	# print('rudder', Fr_car)
	# print('hull', Fh_car)

	F_LRF = Fs_car_LRF + Fh_car_LRF + Fr_car_LRF

	F_surge_car_LRF = np.array([F_LRF[x], 0.0])
	F_sway_car_LRF = np.array([0.0, F_LRF[y]])

	# convert to polar coordinates
	F_surge_pol_LRF = cart2pol(F_surge_car_LRF)
	F_sway_pol_LRF = cart2pol(F_sway_car_LRF)


	F_car_thrust_LRF = np.array([F_LRF[x], 0])
	F_car_thrust_LRF = np.array([F_LRF[x], 0.1*F_LRF[y]])
	#F_car_thrust_LRF = F_LRF# np.array([F_LRF[x], 0])
	#print('Fthrust', F_car_thrust)
	

	# convertto polar coords
	Fpol_thrust_LRF = cart2pol(F_car_thrust_LRF)

	# convert to acceleration by dividing magnitude through by mass
	acceleration_LRF = np.array([Fpol_thrust_LRF[0], Fpol_thrust_LRF[1]/mass])
	# print('acc_LRF_pol', acceleration)
	# print('acc_LRF_car', pol2cart(acceleration))



	# convert to GRF 
	# convert to global coordinates
	# F_surge_pol[0] += boat_angle
	# F_sway_pol[0] += boat_angle
	# acceleration[0] += boat_angle

	# # store data for plotting : LRF (updated to GRF in main code)
	data['surge_force'].append(F_surge_pol_LRF)
	data['sway_force'].append(F_sway_pol_LRF)
	

	# # data['surge_force'].append(Fpol_thrust)
	# 	# change in vel cartesian
	# print('boat_angle', boat_angle)
	# dvdt_pol_GRF = acceleration
	# #print('acc_pol_GRF_', dvdt_pol_GRF)
	# print('acc_pol_GRF_', dvdt_pol_GRF)
	# print('diff', 2 * pi - dvdt_pol_GRF[0])


	# dvdt_car_GRF = pol2cart(dvdt_pol_GRF)
	# print('acc_car_GRF_', dvdt_car_GRF)

	
	#print()

	#print()
	return acceleration_LRF#, F_sway_pol_LRF

# def dvdt_new(v_pol, Fs_pol, Fr_pol, Fh_pol, boat_angle):

# 	"""
# 	Acceleration of the PMW, angle and magnitude, GRF

# 	"""
# 	boat_angle=four_quad(boat_angle)
# 	print(boat_angle)

# 	Fs_car = pol2cart(Fs_pol)
# 	Fr_car = pol2cart(Fr_pol)
# 	Fh_car = pol2cart(Fh_pol)

# 	F = Fs_car + Fh_car + Fr_car
# 	F_surge_car = F * cos(boat_angle)
# 	F_sway_car = F * sin(boat_angle)




#def inertial_moment(F_sway):
	"""
	Find the inertial moment on the boat due to hull resistance to sway force
	Imbalance of bow and stern about the COG causes a turning moment
	"""

	#Fsurge_GRF = data['surge_force'][-1]

	
	# F_sway_pol_LRF = np.array([F_sway_pol_LRF[0] + pi, 
	# 						   F_sway_pol_LRF[1]])

	# inertial force acts in opposition to surge force
	# assume bow inertia is double the stern inertia
	# FI_bow_LRF   = np.array([F_sway[0] + 2*pi, F_sway[1]*2/3])
	# FI_stern_LRF = np.array([F_sway[0] + 2*pi, F_sway[1]*3/3])

	# # moments act at point half the distnace between COG and stern or COG and bow
	# bow_ml = boat_l / 4
	# stern_ml = boat_l / 4

	# # if local sway force is in port (left) direction, Mbow --> clockwise (-ve), Mstern --> anti-clockwise (+ve)
	# # if local sway force is in starboard (right) direction, Mbow --> anti-clockwise (-ve), Mstern --> clockwise (+ve)
	# if 0 <= F_sway[0] < pi:
	# 	Mbow_dir = -1
	# 	Mstern_dir = 1
	# else:
	# 	Mbow_dir = 1
	# 	Mstern_dir = -1

	# M_inertial = (FI_bow_LRF[1] * bow_ml * Mbow_dir +
	# 	          FI_stern_LRF[1] * stern_ml * Mstern_dir)

	# distance from hull COE to 

	#return M_inertial

	

# def dpdt(v_pol, Fs_pol, Fr_pol, Fh_pol, boat_angle):
# 	"""
# 	Velocity of the PMW. 
# 	"""
# 	# # initial vel cartesian
# 	# print('vel_pol_GRF_init_', v_pol)
# 	# v_car = pol2cart(v_pol)
# 	# print('vel_car_GRF_init_', v_car)

# 	# print('using global velocity to find initial local veloity...')
# 	# vel_pol_LRF_init = np.array([v_pol[0]-boat_angle, v_pol[1]])
# 	# print('vel_pol_LRF_init_', vel_pol_LRF_init)
# 	# v_car_LRF_init = pol2cart( vel_pol_LRF_init)
# 	# print('vel_car_LRF_init_', v_car_LRF_init)


# 	# # change in vel cartesian
# 	# dvdt_pol_GRF = dvdt(v_pol, Fs_pol, Fr_pol, Fh_pol, boat_angle)
# 	# print('acc_pol_GRF_', dvdt_pol_GRF)
# 	# dvdt_car_GRF = pol2cart(dvdt_pol_GRF)
# 	# #print()
# 	# print('acc_car_GRF_', dvdt_car_GRF)



# 	# finding velocity in local cartesian frame to prove robot is only travelling in local x direction
# 	# v_car_LRF =  pol2cart(np.array([v_pol[0]-theta, v_pol[1]]))
# 	# # change in vel cartesian
# 	# dvdt_global = dvdt(v_pol, Fs_pol, Fr_pol, Fh_pol, theta)
# 	# local_dvdt = np.array([dvdt_global[0]-theta, dvdt_global[1]])
# 	# print('dvdt_pol_LRF', local_dvdt)
# 	# print('dvdt_car_LRF', pol2cart(local_dvdt))
# 	# dvdt_car_LRF = pol2cart(np.array([dvdt_global[0]-theta, dvdt_global[1]]))
# 	# v_car_LRF = v_car_LRF + dvdt_car_LRF
# 	# print('local_vel_cart', v_car_LRF)


# 	return cart2pol(v_car + dvdt_car_GRF)

# def inertial_moment(Fh_pol, boat_angle):

# 	angle = attack_angle(boat_angle, boat_angle, Fh_pol, rudder=True)
# 	print('angle of attack, boat', angle)

# 	# magnitude of force contributing to rudder moment about boat COG
# 	F_mag = Fh_pol[1] * sin(angle)
# 	print('Fr sin alpha', F_mag)
# 	# rudder force, LRF
# 	Fh_pol_LRF = np.array([four_quad(Fh_pol[0] - boat_angle), Fh_pol[1]])
# 	# angle indicating direction of rudder moment in boat LRF (i.e. pi/2 or -pi/2)
# 	F_ang = moment_force_angle(boat_angle, Fh_pol, boat_angle)
# 	print('F_ang', F_ang)
	
# 	# store hull moment force for pltting
# 	data['hull_moment_force'].append(np.array([F_ang + boat_angle, F_mag]))
# 	#print('rudder_momoent_force', np.array([F_ang, F_mag]))

# 	# moment arm length
# 	moment_arm_l = 0.1#(boat_l / 2 +    # distance rudder hinge to boat COG
# 		              #rudder_l / 2) * cos(abs(rudder_angle - boat_angle))) # distnace rudder hinge to rudder COE with chnage in length due to rudder angle
	
# 	# moment about boat COG
# 	# anti-clockwise is positive direction
# 	M_inertial = F_mag * moment_arm_l * -np.sign(F_ang)

# 	#print('M_inertial', M_inertial)

# 	return M_inertial

# def rudder_moment(Fr_pol, rudder_angle, boat_angle):

# 	angle = attack_angle(rudder_angle, boat_angle, Fr_pol)#, rudder=True)
# 	print('angle of attack, rudder', angle)

# 	# magnitude of force contributing to rudder moment about boat COG
# 	F_mag = Fr_pol[1] * sin(angle)
# 	print('Fr sin alpha', F_mag)
# 	# rudder force, LRF
# 	Fr_pol_LRF = np.array([four_quad(Fr_pol[0] - boat_angle), Fr_pol[1]])
# 	# angle indicating direction of rudder moment in boat LRF (i.e. pi/2 or -pi/2)
# 	F_ang = moment_force_angle(rudder_angle, Fr_pol, boat_angle)
# 	print('F_ang', F_ang)
	
# 	# store rudder moment force for pltting
# 	data['rudder_moment_force'].append(np.array([F_ang + boat_angle, F_mag]))
# 	#print('rudder_momoent_force', np.array([F_ang, F_mag]))
# 	# moment arm length
# 	rudder_ml = (boat_l / 2 +    # distance rudder hinge to boat COG
# 		        (rudder_l / 2) * cos(abs(rudder_angle - boat_angle))) # distnace rudder hinge to rudder COE with chnage in length due to rudder angle
	
# 	# moment about boat COG
# 	# anti-clockwise is positive direction
# 	M_rudder = F_mag * rudder_ml * -np.sign(F_ang)
# 	#print('M_rudder', M_rudder)

# 	return M_rudder

def moment(F_pol, part_angle, boat_angle, moment_arm_length, save_name):

	# angle of attack of force relative to boat
	angle = attack_angle(part_angle, boat_angle, F_pol)
	#print('angle of attack', angle)

	# magnitude of force contributing to rudder moment about boat COG
	F_mag = F_pol[1] * sin(angle)
	#print('Fr sin alpha', F_mag)
	# rudder force, LRF
	#Fr_pol_LRF = np.array([four_quad(F_pol[0] - boat_angle), F_pol[1]])

	# angle indicating direction of rudder moment in boat LRF (i.e. pi/2 or -pi/2)
	F_ang = moment_force_angle(part_angle, F_pol, boat_angle)
	#print('F_ang', F_ang)
	
	# store rudder moment force for pltting
	data[save_name].append(np.array([F_ang + boat_angle, F_mag]))
	#print('rudder_momoent_force', np.array([F_ang, F_mag]))
	# moment arm length
	# rudder_ml = (boat_l / 2 +    # distance rudder hinge to boat COG
	# 	        (rudder_l / 2) * cos(abs(rudder_angle - boat_angle))) # distnace rudder hinge to rudder COE with chnage in length due to rudder angle
	
	# moment about boat COG
	# if COE is further forward/bowward of boat COG (i.e. hull COE): 
	#		pi/2 = +ve (ccw) ,  -pi/2 = -ve (cw)  
	# if COE is further backward/sternward of boat COG (i.e. rudder COE): 
	#		pi/2 = -ve (cw) ,  -pi/2 = +ve (ccw)  
	if np.allclose(four_quad(part_angle), four_quad(boat_angle)): # if hull
		moment_direction = np.sign(F_ang)
	else: # if rudder
		moment_direction = -np.sign(F_ang)



	M = F_mag * moment_arm_length * moment_direction #-np.sign(F_ang)
	#print('M', M)

	return M


def dwdt(Fr_pol, rudder_angle, boat_angle, Fh_pol):
	"""
	Angular velocity of the boat due to the rudder moment
	"""
	# First find rudder moment
    # angle of attack of rudder force to rudder
	# angle = attack_angle(rudder_angle, boat_angle, Fr_pol, rudder=True)
	# print('angle of attack, rudder', angle)

	# # magnitude of force contributing to rudder moment about boat COG
	# F_mag = Fr_pol[1] * sin(angle)
	# print('Fr sin alpha', F_mag)
	# # rudder force, LRF
	# Fr_pol_LRF = np.array([four_quad(Fr_pol[0] - boat_angle), Fr_pol[1]])
	# # angle indicating direction of rudder moment in boat LRF (i.e. pi/2 or -pi/2)
	# F_ang = moment_force_angle(rudder_angle, Fr_pol, boat_angle)
	# print('F_ang', F_ang)
	
	# # store rudder moment force for pltting
	# data['rudder_moment_force'].append(np.array([F_ang + boat_angle, F_mag]))
	# #print('rudder_momoent_force', np.array([F_ang, F_mag]))
	# # moment arm length
	# rudder_ml = (boat_l / 2 +    # distance rudder hinge to boat COG
	# 	        (rudder_l / 2) * cos(abs(rudder_angle - boat_angle))) # distnace rudder hinge to rudder COE with chnage in length due to rudder angle
	
	# # moment about boat COG
	# # anti-clockwise is positive direction
	# M_rudder = F_mag * rudder_ml * -np.sign(F_ang)
	# print('M_rudder', M_rudder)

	rudder_moment_arm = (boat_l / 2 +    # distance rudder hinge to boat COG
		                (rudder_l / 2) * cos(abs(rudder_angle - boat_angle))) # distnace rudder hinge to rudder COE with chnage in length due to rudder angle


	hull_moment_arm = 0.1


	# M_rudder = rudder_moment(Fr_pol, rudder_angle, boat_angle)
	# print('M_rudder', M_rudder)
	M_rudder = moment(Fr_pol, rudder_angle, boat_angle, rudder_moment_arm, save_name='rudder_moment_force')
	

	# M_inertia = inertial_moment(Fh_pol, boat_angle)
	# print('M_inertia', M_inertia)
	M_inertia = moment(Fh_pol, boat_angle, boat_angle, hull_moment_arm, save_name='hull_moment_force')
	print('M_rudder', M_rudder)
	print('M_inertia', M_inertia)

	#print('M_rudder', M_rudder)
	M = M_rudder + M_inertia
	print('M', M)
	#print(np.rad2deg(M))
	#print()

	# print('m_rudder', M_rudder)
	# print('m inertia', M_inertia)


	# second moment of area in yaw axis (Physics of Sailing, By John Kimball, P89)
	# scaler = 5
	# Iyaw = (1/3) * mass * (boat_l/2)**2 * scaler

	# scaler = 5
	Iyaw = (1/3) * mass * (boat_l/2)**2 #* scaler
	#print('I', Iyaw)
	#print(Iyaw)

	# convert to acceleration by dividing moment by mass moment of area
	acc_ang = M / Iyaw
	print('acc_ang', acc_ang)

	return acc_ang

def set_sail_angle(binary_actuator, binary_angles):
	"""
	Adjusts the sail to optimal angle for maximum surge force 
	"""
	# global sa

	# wind_angle = aw_pol[0]

	# if 0 < wind_angle <= pi/2:
	# 	sa = wind_angle - pi/12
	# elif 3*pi/2 < wind_angle <= 2*pi:
	# 	sa = wind_angle + pi/12
	# else:
	# 	sa = wind_angle - pi/2

	global sa

	# apparent wind, LRF
	aw_LRF = aw_pol[0] - theta


	if pi <= aw_LRF <= pi*3/2:
		sa = aw_LRF - pi - pi/12
		print(aw_LRF, "lift sail")

	elif pi/2 <= aw_LRF <= pi:
		sa = aw_LRF + pi + pi/12
		print(aw_LRF, "lift sail")


	else:
		sa = aw_LRF - pi/2
		print(aw_LRF, "drag sail")


	sa = four_quad(sa)

	#print('sa', sa)
	# if sail is binary actuator, use closest avilable sail angle	
	if binary_actuator:
		sa = min(binary_angles, key=lambda x:abs(x-sa))
	#print('sa', sa)

# def dthdt(w, Fr_pol, rudder_angle, theta):
# 	"""
# 	%--------------------------------------------------------------------------
# 	% The angular velocity of the PMW. 
# 	%          out : angular velocity
# 	%--------------------------------------------------------------------------
# 	"""

# 	#print('ang_vel', np.degrees(w + dwdt(Fr_pol, rudder_angle)))
# 	#print('ang_acc', np.degrees(dwdt(Fr_pol, rudder_angle)))
	
# 	return w + dwdt(Fr_pol, rudder_angle, theta)
def save_fig(fig_location, title):
	"""
	Locates the simulation results within local folder 'Projects' and saves figure
	"""
	#plt.savefig(f'{save_location}/r_{round(ra, 3)} s_{round(sa,3)} tw_{round(tw_pol[0],3)}, {round(tw_pol[1],3)}.pdf')
	#plt.savefig(f'{fig_location}/{title}.pdf')
	#plt.savefig(f'{fig_location}/r_{round(ra, 3)} s_{round(sa,3)} tw_{round(tw_pol[0],3)}, {round(tw_pol[1],3)}.pdf')
	plt.savefig(f'{fig_location}/{title}.pdf')
	# for root, dirs, files in os.walk("/Users/hemma/Documents/Projects"):
 #    		for d in dirs:    
 #    			if fnmatch.fnmatch(d, "sailing_simulation_results"):
 #    				save_location = os.path.join(root, d)
 #    				dir_name = time.strftime('%Y-%m-%d--%H-%M-%S')
 #    				os.makedirs(f'{save_location}/{dir_name}', exist_ok=True)
 #    				plt.savefig(f'{save_location}/{dir_name}/r_{round(ra, 3)} s_{round(sa,3)} tw_{round(tw_pol[0],3)}, {round(tw_pol[1],3)}.pdf')
    				#plt.savefig(f'{save_location}/r{ra}_s{sa}_tw{tw_pol}.pdf')


def param_solve(#Z_state, 
				v_pol,
				w,
	            auto_adjust_sail, 
	            binary_actuator, 
	            binary_angles,
	            plot_force_coefficients):

	# global sa, ra, vpol, pos_pol, aw_pol, theta, w
	global vpol, aw_pol, start_time
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

    # give each state variable a name
	#pos_pol = Z_state[0]
	#v_pol =   Z_state[0]#[1]
	#theta =   Z_state[2]
	#w =       Z_state[1]#[3]

	#print('twpol', tw_pol)
	aw_pol = appWind(tw_pol, v_pol)
	data["true_wind"].append(tw_pol)
	data["apparent_wind"].append(aw_pol)

	#print(aw_pol)

	if auto_adjust_sail:
		time_now = t 
		if time_now - start_time >= L:
			print('adjusting sails, t= ', t)
			set_sail_angle(binary_actuator, binary_angles)
			start_time = t
			


	#print(sa)

	vw_pol = np.array([four_quad(v_pol[0]+pi), 
		                         v_pol[1]])
	
	# calculate lift and drag force
	Ls_pol = aero_force(part='sail',   force='lift', apparent_fluid_velocity=aw_pol, part_angle=sa, boat_angle=theta, plot_coefficients=plot_force_coefficients)
	Ds_pol = aero_force(part='sail',   force='drag', apparent_fluid_velocity=aw_pol, part_angle=sa, boat_angle=theta, plot_coefficients=plot_force_coefficients)
	Lr_pol = aero_force(part='rudder', force='lift', apparent_fluid_velocity=vw_pol, part_angle=ra, boat_angle=theta, plot_coefficients=plot_force_coefficients)  
	Dr_pol = aero_force(part='rudder', force='drag', apparent_fluid_velocity=vw_pol, part_angle=ra, boat_angle=theta, plot_coefficients=plot_force_coefficients) 
	Lh_pol = aero_force(part='hull',   force='lift', apparent_fluid_velocity=vw_pol, part_angle=theta, boat_angle=theta, plot_coefficients=plot_force_coefficients)  
	Dh_pol = aero_force(part='hull',   force='drag', apparent_fluid_velocity=vw_pol, part_angle=theta, boat_angle=theta, plot_coefficients=plot_force_coefficients) 

	# print('rudder_lift', pol2cart(Lr_pol))
	# print('rudder_drag', pol2cart(Dr_pol))


	# resolve into forces on boat
	Fs_pol = sumAeroVectors(Ls_pol, Ds_pol, 'sail')  
	Fr_pol = sumAeroVectors(Lr_pol, Dr_pol, 'rudder')
	#print('rudder_force', pol2cart(Fr_pol))
	#print('saved_rudder_force', Fr_pol)
	
	#print()
	#print('Fr_pol', Fr_pol)
	Fh_pol = sumAeroVectors(Lh_pol, Dh_pol, 'hull')


	# save angle
	data["sail_angle"].append(sa)
	data["rudder_angle"].append(ra)
	data["sail_area"].append(A_s)
	# save lift force
	data['sail_lift'].append(Ls_pol)
	data['rudder_lift'].append(Lr_pol) 
	data['hull_lift'].append(Lh_pol) 
	# save drag force
	data['sail_drag'].append(Ds_pol)
	data['rudder_drag'].append(Dr_pol)
	data['hull_drag'].append(Dh_pol)
	# save force
	data['sail_force'].append(Fs_pol)
	# print(data['sail_force'][-1])
	# print()
	data['rudder_force'].append(Fr_pol)
	#print(data['rudder_force'][-1])
	#print()
	data['hull_force'].append(Fh_pol)


	# rate of change of model parameters
	# dZdt = [#dpdt(v_pol, Fs_pol, Fr_pol, Fh_pol, theta), 
	#         dvdt(v_pol, Fs_pol, Fr_pol, Fh_pol, theta), 
	#         #dthdt(w, Fr_pol, ra, theta),  
	#         dwdt(Fr_pol, ra, theta),
	# 		]
	acceleration = dvdt(Fs_pol, Fr_pol, Fh_pol, theta)
	#print('acc_LRF_pol', acceleration)

	ang_acceleration =  dwdt(Fr_pol, ra, theta, Fh_pol)

			
	#return np.array([acceleration, ang_acceleration])
	return acceleration, ang_acceleration
	#return np.array(dZdt)
  

# MAIN PROGRAM

bin_angles = [0.0, 0.19, 0.77, 0.86, 1.05, 1.05, 1.33, 1.57, 1.57, 1.81, 2.09, 2.09, 2.28, 2.38, 2.95, 3.14]

steps = 10

def main(rudder_angle = 0 , 
		 sail_angle = pi/6,
		 auto_adjust_sail = False,
		 Time = np.arange(steps),
		 time_ticks = np.arange(steps),
		 # true_wind_polar = np.array([pi - (pi/6), 5]),
		 true_wind_polar = [np.array([pi - (pi/6), 5])] * steps,
		 binary_actuator = False,
		 binary_angles = bin_angles,
		 save_figs = False,
		 fig_location = save_location,
		 plot_force_coefficients = True,
		 output_plot_title = None,
		 latency = 0 # seconds
		 ):

	"""
	Main program.

	Takes optional inputs for true wind angle, sail angle and rudder angle
	"""

	global rho_air, rho_water
	# boat geometry
	global mass, boat_l, boat_w, rudder_l, sail_l
	global A_s, A_r, A_h, ARs, ARr, ARh
	global c_s, c_r, c_h, t_s, t_r, t_h, ra, sa
	global CN1inf_s_max, CN1inf_r_max, CN1inf_h_max, CN1inf_s_min, CN1inf_r_min, CN1inf_h_min
	global CD0_s, CD0_r, CD0_h
	global ACL1_inf_s, ACL1_inf_r, ACL1_inf_h
	global hull_drag_scale_factor, hull_pre_stall_drag_scale_factor, hull_pre_stall_scale_factor, sail_drag_scale_factor, rudder_scale_factor
	# initial conditions
	global pos_car, pos_pol, v_car, v_pol, theta, w
	global Z_init_state, data
	global tw_pol
	global x, y
	global t, start_time, time_now
	global L # latency

	x, y = 0, 1

	L = latency

	rho_air = 1.225;   # density air
	rho_water = 1000;  # density water

	# boat mass
	mass = 20# 50#10 # kg

	# boat geometry
	boat_l = 1         # boat length
	boat_w = 0.5       # boat width
	rudder_l = 0.4     # rudder length
	sail_l = 0.8       # sail length

	# sail / rudder area
	A_s = 0.64         # sail area
	A_r = 0.01# 0.02# 0.05          # rudder area
	A_h = 0.1          # hull area 

	# aspect ratio
	ARs = 4            # aspect ratio sail
	ARr = 4            # aspect ratio rudder
	ARh = 4            # aspect ratio hull

	# chord
	c_s= sail_l        # sail chord, m
	c_r= rudder_l      # rudder chord, m
	c_h= boat_l        # hull chord, m

	# thickness
	t_s = 0.15 * sail_l      # sail thickness
	t_r = 0.15 * rudder_l      # rudder thickness
	t_h = 0.5 * boat_l      # hull thickness

	# Maximum normal coefficient for an infinite foil
	CN1inf_s_max = 1 # empirical data from Fage and Johannsen, 1876 = 0.45
	CN1inf_r_max = 0.1
	CN1inf_h_max = 0.1

	# Minimum  normal coefficient (i.e. at angle of attack = A0) for an infinite foil
	CN1inf_s_min = 0 # empirical data from Johannsen = 0
	CN1inf_r_min = 0
	CN1inf_h_min = 0

	ACL1_inf_s = 9         # α at max pre-stall lift (deg), empirical data (Fage and Johansen, 1876)
	ACL1_inf_r = 9# 45         # α at max pre-stall lift (deg), empirical data (Fage and Johansen, 1876)
	ACL1_inf_h = 9         # α at max pre-stall lift (deg), empirical data (Fage and Johansen, 1876)

	# Minimum drag coefficient (i.e. CD at angle of attack = A0) for an infinite foil
	CD0_r = 0
	CD0_s = 0
	CD0_h = 0.001

	# intial conditions
	pos_car = np.array([0, 0])      # boat position, GRF (global reference frame)
	pos_pol = cart2pol(pos_car)
	v_car = np.array([0, 0])        # boat velocity (x, y), GRF
	v_pol = cart2pol(v_car)
	theta = 0;                      # boat angle, GRF
	w = 0;                          # boat angular velocity, GRF
	# tw_pol = np.array([pi + (pi/6), 5])   # true wind velocity (angle, magnitude), GRF
	# tw_pol = np.array([2*pi - deg2rad(45) , 5])
	# tw_pol = np.array([pi - deg2rad(25) , 5])
	# tw_pol = true_wind_polar #np.array([2*pi , 5])

	ra = rudder_angle #pi/10                          # rudder angle, LRF (local reference frame)
	sa = sail_angle   #pi/2                       # sail angle, LRF

	hull_drag_scale_factor = 0.01
	hull_pre_stall_drag_scale_factor = 0.1
	hull_pre_stall_scale_factor = 8
	rudder_scale_factor = 10
	sail_drag_scale_factor = 1#0.5#0.1


	# initial values of Time-varying parameters
	Z_init_state = [#pos_pol, 
					v_pol,
					#theta,
					w] 

	# Time = np.arange(0, 20, 1)
	# Time = np.arange(10)

	# check the number of wind coordinates given is the same as the number of timesteps
	if len(true_wind_polar) != len(Time):
		# make at least as many wind sata points as timesteps
		true_wind_polar *= int(np.ceil(len(Time) / len(true_wind_polar)))
		# crop list to same length as no of time steps
		true_wind_polar = true_wind_polar[:len(Time)]


	#sail_angle, rudder_angle, sail_area, position, velocity, heading, angular_vel = [], [], [], [], [], [], []
	data = {'position' : [],    'true_wind' : [],    'apparent_wind' : [],    
	        'velocity' : [],    'angular_vel' : [],  'sail_area' : [],
	        'sail_angle' : [],  'rudder_angle' : [], 'heading' : [],        
	        'sail_lift' : [],   'rudder_lift' : [],  'hull_lift' : [], 
	        'sail_drag' : [],   'rudder_drag' : [],  'hull_drag' : [],
	        'sail_force' : [],  'rudder_force' : [], 'hull_force' : [],
	        'surge_force' : [], 'sway_force' : [],   
	        'rudder_moment_force' : [], 'hull_moment_force' : [],
	        }


	# data["sail_angle"].append(sa)
	# data["rudder_angle"].append(ra)


	
	start_time = Time[0]

	# solve parameters at each Timestep
	for t, tw_pol in zip (Time, true_wind_polar):
		print()
		print(t)
		#print()
		# data["sail_angle"].append(sa)
		# data["rudder_angle"].append(ra)
		#print('boat_pos', pol2cart(pos_pol))#Z_init_state[0]))
		data["position"].append(pos_pol)#Z_init_state[0])
		data["velocity"].append(v_pol)#Z_init_state[0])
		data["heading"].append(theta)#Z_init_state[2])	
		data["angular_vel"].append(w)#Z_init_state[1])

		# print('velocity_pol_start', data["velocity"][t])
		# print('position_pol_start', data["position"][t])
		# print()
		# print('velocity_cart_start', pol2cart(data["velocity"][t]))
		# print('position_cart_start', pol2cart(data["position"][t]))
		# print()

		# find rates of change
		# state = param_solve(#Z_init_state, 
		# 					v_pol,
		# 					w,
		# 	                auto_adjust_sail, 
		# 	                binary_actuator, 
		# 	                binary_angles, 
		# 	                plot_force_coefficients)


		# LOCAL ACCEERATION AND ANGULAR ACCELERATION
		acc_LRF, ang_acc = param_solve(#Z_init_state, 
							v_pol,
							w,
			                auto_adjust_sail, 
			                binary_actuator, 
			                binary_angles, 
			                plot_force_coefficients)

		print('acc_LRF_pol', acc_LRF)

		# convert global velocity to local velocity
		v_pol_LRF = np.array( [v_pol[0]-theta, v_pol[1]] )
		print('v_pol_LRF_init', v_pol_LRF)

		# update angular velocity and angle
		w += ang_acc
		theta += w
		print('theta', theta)

		# # convert global velocity to local velocity
		# v_pol_LRF = np.array( [v_pol[0]-theta, v_pol[1]] )
		# print('v_pol_LRF_init', v_pol_LRF)

		# update local velocity
		v_pol_LRF = cart2pol( pol2cart(v_pol_LRF) + pol2cart(acc_LRF) )
		print('v_pol_LRF', v_pol_LRF)
		print('v_cart_LRF', pol2cart(v_pol_LRF))

		# convert back to global velocity to calculate new forces
		v_pol = np.array( [v_pol_LRF[0]+theta, v_pol_LRF[1]] )
		print('v_pol_GRF', v_pol)

		pos_pol = cart2pol( pol2cart(pos_pol) + pol2cart(v_pol) )
		print('pos_pol_GRF', pos_pol)
		

		# # update global position
		# print('pos_pol', pos_pol)




		# # convert acceleration to GRF using NEW angle
		# acc_GRF = np.array( [acc_LRF[0]+theta, acc_LRF[1]] )
		# acc_car_GRF = pol2cart(acc_GRF)
		# print('acc_GRF_pol', acc_GRF)
		# print('acc_GRF_car', acc_car_GRF)

		# # update velocity and position
		# print('v_init', v_pol)
		# v_pol = cart2pol( pol2cart(v_pol) + pol2cart(acc_GRF) )
		# print('v_pol', v_pol)
		# pos_pol = cart2pol( pol2cart(pos_pol) + pol2cart(v_pol) )
		# print('pos_pol', pos_pol)


		# F_surge_pol[0] += boat_angle
		# F_sway_pol[0] += boat_angle
		# acceleration[0] += boat_angle

		# Convert to GRF
		data['surge_force'][-1][0] += theta
		data['sway_force'][-1][0] += theta

		#return data
		

		# # data['surge_force'].append(Fpol_thrust)
		# 	# change in vel cartesian
		# print('boat_angle', boat_angle)
		# dvdt_pol_GRF = acceleration
		# #print('acc_pol_GRF_', dvdt_pol_GRF)
		# print('acc_pol_GRF_', dvdt_pol_GRF)
		# print('diff', 2 * pi - dvdt_pol_GRF[0])


		# dvdt_car_GRF = pol2cart(dvdt_pol_GRF)
		# print('acc_car_GRF_', dvdt_car_GRF)



		#print('v_cart_init', pol2cart(v_pol))

		#v_pol = cart2pol( pol2cart(v_pol) + pol2cart(acc) )
		#print('v_pol', v_pol)
		#v_cart = pol2cart(v_pol)
		#print('v_cart', v_cart)
		#v_pol = cart2pol(v_cart)
		# print('v_pol', v_pol)
		# print('v_car_local', pol2cart(np.array([v_pol[0]+theta, v_pol[1]])))
		# #print('v_pol', v_pol)
		# print()
		#w += ang_acc


		# for i, (z, s) in enumerate(zip(Z_init_state, state)):
		# 	# array values
		# 	if type(z) == np.ndarray:
		# 		Z_init_state[i] = cart2pol(pol2cart(z)+pol2cart(s))
		# 	# single variables
		# 	else:
		# 		Z_init_state[i] = z + s
		
		#pos_pol = cart2pol( pol2cart(pos_pol) + pol2cart(v_pol) )
		#theta += w
		# print('velocity_cart_end', pol2cart(Z_init_state[1]))
		# print('position_cart_end', pol2cart(Z_init_state[0]))
		# print()
		# print()
		# print()





	global ax1
	# plot all the data
	fig1, ax1 = plt.subplots()
	for i in Time:

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
	print("im here")	 
	if save_figs:
		save_fig(fig_location, title)
	else:
		plt.show()

if __name__ == '__main__': main()

	







