import numpy as np
from numpy import pi

x, y = 0, 1


def cart2pol(coords):
	x = coords[0]
	y = coords[1]
	rho = np.sqrt(x**2 + y**2)
	phi = np.arctan2(y, x)
	if phi < 0:
		phi += 2*pi
		#return(rho, phi)
	return np.array([rho, phi])


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

rho_air = 1.225;
rho_water = 1000;

# sail / rudder parameters
A_s = 0.25;
A_r = 0.05;
# aspect ratio
ARs = 5
ARr = 0.5

# intial conditions
pos = np.array([0, 0])
v_car = np.array([0, 0])
v_pol = cart2pol(v_car)
print('vpol', v_pol)
theta = 0;
w = 0;
Z_init = [pos, theta, v_car, w]

# time 
tspan = [0, 1];

# wind
tw_pol = np.array([pi, 3])


# rudder angle if you are standing on boat
# % -ve towards port (left)
# % +ve towards starboard (right)
dr = 0

# sail angle if you are standing on boat
# angle of stern-ward end of sail to follow RH coord system
ds = pi;   





def appWind():
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

	tw_car = pol2cart(tw_pol)
	#print('tw_car', tw_car)

	aw_car = np.array([(tw_car[x] - v_car[x]), 
					   (tw_car[y] - v_car[y])]) 

	aw_pol = cart2pol(aw_car)

	return aw_car, aw_pol

def attack_angle(area, v_fluid_car, plane_angle):
	"""
	% - inputs:
	%          A : plane area of sail or wing
	%          vfluid....................... 2x1 array
	%          v_fluid : fluid velocity, manitude and angle relative to PMW
	%          frame
	%          d : sail or rudder angle      
	% - output:
	%          alpha : the smallest angle between the two vectors
	"""
	d = plane_angle
	plane_car = pol2cart([d, 1])

	U = v_fluid_car
	if np.array_equal(U, np.array([0.0, 0.0])):  # if fluid (i.e. boat) not moving
		alpha = 0 				# angle of attack defaults to 0
	else:
		V = plane_car

		print(U)
		print(V)
		print(np.linalg.norm(U))
		print(np.linalg.norm(V))
		cosalpha = np.dot(U, V) / np.dot(np.linalg.norm(U), np.linalg.norm(V))
		print("cosalpha", cosalpha)

		alpha = np.arccos(cosalpha)

	print("alpha", alpha)

	return alpha

def lift_angle(area, v_fluid_pol, plane_angle):
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
	
	# cartesian coords in boat frame of ref (0 rads : bow of boat)	
	d = plane_angle
	plane_car = pol2cart([d - theta, 1])
	fluid_ang = v_fluid_pol[0] - theta
	v_fluid_car = pol2cart([fluid_ang, v_fluid_pol[1]])

	# absolute angle of sail or rudder as a refernce angle for computing 
	# 1. angle of attack  	
	# 2. direction of lift     
	p = np.arctan(abs(plane_car[y])/ 
		     abs(plane_car[x]))


	#establish orientation or sail or rudder            
	if (plane_car[x] / plane_car[y]) < 0:	# 2nd or 4th quadrant 
	    
	    if ((2 * pi - fluid_ang > fluid_ang > 3 * pi/2 - p) or 
	    	(pi - p             > fluid_ang > pi/2 - p)):

	    	la = fluid_ang - pi/2
	    else:
	    	la = fluid_ang + pi/2
    
	else:	# 1st or 3rd quadrant
	    if (p               < fluid_ang   <  fluid_ang +  pi/2 or 
	    	fluid_ang + pi  < fluid_ang   <  3 * pi/2  + fluid_ang):

	    	la = fluid_ang + pi/2
	    else:
	        la = fluid_ang - pi/2

	# convert angle back to global refernce frame            
	return la + theta

def aero_force(part, force):
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
	    A = A_s
	    v_fluid_pol = aw_pol
	    v_fluid_car = aw_car
	    d = ds
	    rho = rho_air
	    AR = ARs
	    
	else: # part == 'rudder'
	    A = A_r
	    v_fluid_pol = v_pol
	    v_fluid_car = v_car
	    d = dr
	    rho = rho_water
	    AR = ARr
	
	# angle of attack    
	alpha = attack_angle(area=A, v_fluid_car=v_fluid_car, plane_angle=d)

	# lift coefficient
	if alpha < pi/6:
	    CL = (9 / pi) * alpha 
	else:
	    CL = -(4.5/pi) * alpha + (4.5/2)

	# drag coefficient
	CD = 0.05 + CL**2 / (pi * AR);

	# aero force
	if force == 'lift':
		lift_a = lift_angle(area=A, v_fluid_pol=v_fluid_pol, plane_angle=d)
		lift_force = 0.5 * rho * A * v_fluid_pol[1]**2 * CL
		return np.array([lift_a, lift_force])

	else: # force = 'drag'
		drag_angle = v_fluid_pol[0];
		drag_force = 0.5 * rho * A * v_fluid_pol[1]**2 * CD
		return np.array([drag_angle, drag_force])
                                   
# apparent wind
aw_car, aw_pol = appWind()

L_s = aero_force(part='sail', force='lift')
D_s = aero_force(part='sail', force='drag')
L_r = aero_force(part='rudder', force='lift')  
D_r = aero_force(part='rudder', force='drag') 

# print(f'{L_s}/n{D_s}/n{L_r}/n{D_r}')
print(L_s)
print(D_s)
print(L_r)
print(D_r)







