
"""
Sailing Model for Portugese Man of War Robot

Inputs:
- Wind angle and force (global frame)
- Sail angle (local frame)
- Sail aspect ratio as function of sail angle (to model soft robotic PMW-inspired sail)
- Rudder angle (local frame)

Assumptions:
- All hydronamic side force is translated into lateral motion (no heeling moment)
- Hull shape behaves as aerofoil 
- maximum normal force coefficent of 1 used to achieve characteristic post-stall lift and drag curves (empirical value 0.45)

TODO
- plot sail area, chord, thicknes and max nornmal force coefficent should chnage dynamically with sail angle --> examine resulting drag coefficient
- use odeint solver with empirical wind data as forcing function
- only solve acceleration at each timestep - position can be derived
- find slope of linear segment of pre-stall lift (simplified) more accuratelty
- characterise real airfoil and subs in drag coefficent curve for estmaited curve in this model
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

x, y = 0, 1

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


rho_air = 1.225;   # density air
rho_water = 1000;  # density water

# boat geometry
boat_l = 1         # boat length
boat_w = 0.5       # boat width
rudder_l = 0.4     # rudder length
sail_l = 0.8       # sail length

# sail / rudder area
A_s = 0.64         # sail area
A_r = 0.05          # rudder area
A_h = 0.1          # hull area 

# aspect ratio
ARs = 4            # aspect ratio sail
ARr = 2            # aspect ratio rudder
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

# Minimum drag coefficient (i.e. CD at angle of attack = A0) for an infinite foil
CD0_r = 0
CD0_s = 0
CD0_h = 0.01

# intial conditions
pos_car = np.array([0, 0])      # boat position, GRF (global reference frame)
pos_pol = cart2pol(pos_car)
v_car = np.array([0, 0])        # boat velocity (x, y), GRF
v_pol = cart2pol(v_car)
theta = 0;                      # boat angle, GRF
w = 0;                          # boat angular velocity, GRF
tw_pol = np.array([pi + (pi/6), 5])   # true wind velocity (angle, magnitude), GRF
tw_pol = np.array([2*pi - deg2rad(45) , 5])
ra = 0                          # rudder angle, LRF (local reference frame)
sa = pi/4                       # sail angle, LRF


# initial values of time-varying parameters
Z_init_state = [pos_pol, 
				v_pol,
				theta,
				w] 





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





def attack_angle(part_angle, boat_angle, incident_vector_polar):
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

		# convert angles to global frame
		part_angle += boat_angle

		# check angle still expressed in 4 quadrants
		part_angle = four_quad(part_angle)

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


def lift_angle(part_angle, incident_vector_polar):
	"""
	Returns the angle of the lift force on a component, expressed in GRF
	"""
	
	
	dummy_len = 1.   # dummy cartesian coords in local boat frame of ref 
	# Find minimum angle of part relative to boat (i.e. LRF)
	pa_car = pol2cart([part_angle, dummy_len])
	# Find minimum ABSOLLUTE angle of part relative to boat (i.e. LRF)
	pa_abs= np.arctan(abs(pa_car[y])/ 
		     	      abs(pa_car[x]))

	
	# Incident vector angle angle, LRF
	V_pol = incident_vector_polar 
	fab = V_pol[0] - theta
	fab = four_quad(fab)


	# Establish orientation or sail or rudder  
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

	# convert angle to global refernce frame            
	la += theta
	la = four_quad(la)
	return la 

def aero_coeffs(attack_angle, AR, c, t, CN1inf_max, CD0, part):  
	"""
	Computes the lift and drag coefficient for a given angle of attack.
	Considers pre and post stall condition up to angle of attack = 90 degrees to incident surface
    Apply drag coefficient scaling factor to make sure CD > CL for hull and rudder
    Uses model from "Models of Lift and Drag Coefficients of Stalled and Unstalled Airfoils in Wind Turbines and Wind Tunnels" by David A. Spera
	"""
	# David A. Spera model uses degrees
	a = rad2deg(attack_angle)    # angle of attack

	############################
	# David A. Spera Model 
	############################
	# Parameters for infinite foil
	# empirical data (Fage and Johansen, 1876)
	# α = angle of attack
	A0 = 0               # α, where CL = 0, 
	ACL1_inf = 9         # α at max pre-stall lift (deg), empirical data (Fage and Johansen, 1876)
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
		CL1 = S1 * (a - A0) + RCL1 * ((a - A0)/(ACL1 - A0))**N1


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
	if (2*A0 <= a < ACL1):
		CD2 = 0
	elif ACL1 <= a:
		ang = ((a - ACD1)/(90.0 - ACD1)) * 90 
		CD2 = CD1max + (CD2max - CD1max) * sin(deg2rad(ang))	

	CL = max(CL1, CL2)

	#print(CD1, CD2)
	CD = max(CD1, CD2)

	return CL, CD


def aero_force(part, force, apparent_fluid_velocity, part_angle, boat_angle):
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
	attack_a = np.linspace(0, pi, 1000)
	cl, cd = [], []
	for a in attack_a:
		CL, CD = aero_coeffs(a, AR, c, t, CN1inf_max, CD0, part)  				 
		cl.append(CL)
		cd.append(CD)
	#fig = plt.subplots()
	if part == 'hull':
		attack_a= rad2deg(attack_a)
		plt.plot(attack_a, cl, label='lift '+ part)
		plt.plot(attack_a, cd, label='drag '+ part)
		plt.legend('upper center', bbox_to_anchor=(0.5,-0.1))
		plt.title(part)
		plt.xlim((0, 90))
		plt.ylim(ymin=0)


	# find lift and drag coefficent 
	CL, CD = aero_coeffs(alpha, AR, c, t, CN1inf_max, CD0, part)

	if force == 'lift':
		C = CL
		angle = four_quad(lift_angle(part_angle, incident_vector_polar = V_pol))
	else: # force == 'drag'
		C = CD
		angle = four_quad(V_pol[0])
		#angle = V_pol[0]

	force = 0.5 * rho * A * V_pol[1]**2 * C		
	return np.array([angle, force])


def sumAeroVectors(lift_pol, drag_pol):
	"""
	Returns resultant of perpendicular lift or drag forces as polar coordinates in GRF
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

	"""
	Draw velocity and force vectors
	"""

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
                   #[pos_car[x],          pos_car[y], Ls_car[x], Ls_car[y], 'Lsail'],
                   #[pos_car[x],          pos_car[y], Ds_car[x], Ds_car[y], 'Dsail'],
                   #[pos_car[x]-boat_l/2, pos_car[y], Lr_car[x], Lr_car[y], 'Lrud'],
                   #[pos_car[x]-boat_l/2, pos_car[y], Dr_car[x], Dr_car[y], 'Drud'],
                   #[pos_car[x], pos_car[y], Lh_car[x], Lh_car[y], 'Lhull'],
                   #[pos_car[x], pos_car[y], Dh_car[x], Dh_car[y], 'Dhull'],
                   #[pos_car[x], pos_car[y], v_car[x],  v_car[y], 'v'], # sail lift   
                   [pos_car[x], pos_car[y], aw_car[x],  aw_car[y], 'aw'],          
                   [pos_car[x], pos_car[y], tw_car[x],  tw_car[y], 'tw'],
                   [pos_car[x],             pos_car[y], Fs_car[x],  Fs_car[y], 'Fs'],
                   [pos_car[x],             pos_car[y], Fh_car[x],  Fh_car[y], 'Fh'],
                   #[pos_car[x]-boat_l/2   , pos_car[y], Fr_car[x],  Fr_car[y], 'Fr']
                   ]


	colors = cm.rainbow(np.linspace(0, 1, len(vectors)))

	#for n, (V, c, label) in enumerate(zip(vectors, colors, labels), 1):
	for n, (V, c) in enumerate(zip(vectors, colors), 1):
		# ax1.quiver(V[0], V[1], V[2], V[3], color=c, scale=5)
		quiver_scale = 40
		Q = plt.quiver(V[0], V[1], V[2], V[3], color=c, scale=quiver_scale)
		#plt.quiverkey(Q, -1.5, n/2-2, 0.25, label, coordinates='data')
		quiver_key_scale = quiver_scale/10#100
		plt.quiverkey(Q, 1.05 , 1.1-0.1*n, quiver_key_scale, V[4], coordinates='axes')


	# #ax1.autoscale(enable=True, axis='both', tight=None)
	# ax1.set_xlim([0, 4])
	# ax1.set_ylim([0, 4])

def dvdt(v_pol, Fs_pol, Fr_pol, Fh_pol, theta):
	"""
	Acceleration of the PMW, angle and magnitude, GRF
	"""
	# Sail, rudder, hull, forces in LRF
	Fs_pol[0] -= theta 
	Fr_pol[0] -= theta 
	Fh_pol[0] -= theta

	# convert to cartesian coords
	Fs_car = pol2cart(Fs_pol)
	Fr_car = pol2cart(Fr_pol)
	Fh_car = pol2cart(Fh_pol)
	v_car = pol2cart(v_pol)

	thrust = Fs_car[x] + Fh_car[x]
	side_force = Fs_car[y] + Fh_car[y] #+ hull_side_resistance

	# F_car = np.array([thrust, side_force])

	# # convert to polar coordinates 
	# Fpol = cart2pol(F_car)
	# # convert angle to GRF
	# Fpol[0] += theta

	# ignoring side force on boat
	F_car_thrust = np.array([thrust, 0])
	# convert to polar coordinates
	Fpol_thrust = cart2pol(F_car_thrust)
	# convert angle to GRF
	Fpol_thrust[0] += theta
	# convert to acceleration by dividing by mass
	mass = 10
	
	# acceleration = np.array([Fpol[0], 
	# 	                     Fpol[1]/mass])
	acceleration = np.array([Fpol_thrust[0], 
	                         Fpol_thrust[1]/mass])
	return acceleration


def dpdt(v_pol, Fs_pol, Fr_pol, Fh_pol, theta):
	"""
	Velocity of the PMW. 
	"""

	# current vel cartesian
	v_car = pol2cart(v_pol)

	# change in vel cartesian
	dvdt_car = pol2cart(dvdt(v_pol, Fs_pol, Fr_pol, Fh_pol, theta))

	print('dvdt_car', dvdt_car)
	print('dpdt_car', (v_car + dvdt_car))
	print()

	return cart2pol(v_car + dvdt_car)


def dwdt(w, Fs_pol, Fr_pol, theta, rudder_a):
	"""
	Angular velocity of the boat due to the rudder moment
	"""

	# Sail and drag forces in boat frame of ref
	Fs_pol[0] -= theta 
	Fr_pol[0] -= theta 

	# convert to cartesian coords
	Fs_car = pol2cart(Fs_pol)
	Fr_car = pol2cart(Fr_pol)
	#Fs_car, Fr_car = force_wrt_boat(Fs_pol, Fr_pol, theta)

	return 0

	# return (Fr_car [1] * 
	# 	   (boat_l/2 + abs(rudder_l * cos(rudder_a))) + # moment due to force on rudder
	# 	   rotation_drag(w))							  # moment due to rotational drag



def dthdt(w, Fs_pol, Fr_pol, theta, d_rudder):
	"""
	%--------------------------------------------------------------------------
	% The angular velocity of the PMW. 
	%          out : angular velocity
	%--------------------------------------------------------------------------
	"""
	#return w + dwdt(w, Fs_pol, Fr_pol, theta, d_rudder)

	# Sail and drag forces in boat frame of ref
	#Fs_pol[0] -= theta 
	Fr_pol[0] -= theta 

	# convert to cartesian coords
	#Fs_car = pol2cart(Fs_pol)
	Fr_car = pol2cart(Fr_pol)
	#Fs_car, Fr_car = force_wrt_boat(Fs_pol, Fr_pol, theta)



	return 0


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
	global aw_pol

    # give each state variable a name
	pos_pol = Z_state[0]
	v_pol =   Z_state[1]
	theta =   Z_state[2]
	w =       Z_state[3]

	aw_pol = appWind(tw_pol, v_pol)
	data["apparent_wind"].append(aw_pol)

	vw_pol = np.array([four_quad(v_pol[0]+pi), 
		                         v_pol[1]])
	
	# calculate lift and drag force
	Ls_pol = aero_force(part='sail',   force='lift', apparent_fluid_velocity=aw_pol, part_angle=sa, boat_angle=theta)
	Ds_pol = aero_force(part='sail',   force='drag', apparent_fluid_velocity=aw_pol, part_angle=sa, boat_angle=theta)
	Lr_pol = aero_force(part='rudder', force='lift', apparent_fluid_velocity=vw_pol, part_angle=ra, boat_angle=theta)  
	Dr_pol = aero_force(part='rudder', force='drag', apparent_fluid_velocity=vw_pol, part_angle=ra, boat_angle=theta) 
	Lh_pol = aero_force(part='hull',   force='lift', apparent_fluid_velocity=vw_pol, part_angle=theta, boat_angle=theta)  
	Dh_pol = aero_force(part='hull',   force='drag', apparent_fluid_velocity=vw_pol, part_angle=theta, boat_angle=theta) 

	# resolve into forces on boat
	Fs_pol = sumAeroVectors(Ls_pol, Ds_pol)  
	Fr_pol = sumAeroVectors(Lr_pol, Dr_pol)
	Fh_pol = sumAeroVectors(Lh_pol, Dh_pol)


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
	data['rudder_force'].append(Fr_pol)
	data['hull_force'].append(Fh_pol)


	# rate of change of model parameters
	dZdt = [dpdt(v_pol, Fs_pol, Fr_pol, Fh_pol, theta), 
	        dvdt(v_pol, Fs_pol, Fr_pol, Fh_pol, theta), 
	        dthdt(w, Fs_pol, Fr_pol, theta, ra),  
	        dwdt(w, Fs_pol, Fr_pol, theta, ra),
			]

	return np.array(dZdt)
  

# MAIN PROGRAM
time = np.arange(0, 20, 1)
time = np.arange(5)

#sail_angle, rudder_angle, sail_area, position, velocity, heading, angular_vel = [], [], [], [], [], [], []
data = {'position' : [],    'apparent_wind' : [],    
        'velocity' : [],    'angular_vel' : [],  'sail_area' : [],
        'sail_angle' : [],  'rudder_angle' : [], 'heading' : [],        
        'sail_lift' : [],   'rudder_lift' : [],  'hull_lift' : [], 
        'sail_drag' : [],   'rudder_drag' : [],  'hull_drag' : [],
        'sail_force' : [],  'rudder_force' : [], 'hull_force' : [],
        }


# data["sail_angle"].append(sa)
# data["rudder_angle"].append(ra)


# solve parameters at each timestep
for t in time:
	# data["sail_angle"].append(sa)
	# data["rudder_angle"].append(ra)	
	data["position"].append(Z_init_state[0])
	data["velocity"].append(Z_init_state[1])
	data["heading"].append(Z_init_state[2])	
	data["angular_vel"].append(Z_init_state[3])

	print('velocity_pol_start', data["velocity"][t])
	print('position_pol_start', data["position"][t])
	print()
	print('velocity_cart_start', pol2cart(data["velocity"][t]))
	print('position_cart_start', pol2cart(data["position"][t]))
	print()

	# find rates of change
	state = param_solve(Z_init_state)

    # update parameters
	for i, (z, s) in enumerate(zip(Z_init_state, state)):
		# array values
		if type(z) == np.ndarray:
			Z_init_state[i] = cart2pol(pol2cart(z)+pol2cart(s))
		# single variables
		else:
			Z_init_state[i] = z + s

	print('velocity_cart_end', pol2cart(Z_init_state[1]))
	print('position_cart_end', pol2cart(Z_init_state[0]))
	print()
	print()
	print()


# plot all the data
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

plt.show()

	







