import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.lines as mlines
import matplotlib.cm as cm

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

# boat parameters
boat_l = 1
boat_w = 0.5
rudder_l = 0.2
sail_l = 0.8
# points_x = [l/2, l/2, -l/2, -l/2, l/2]
# points_y = [-w/2, w/2, w/2, -w/2, -w/2]
# points = np.array([points_x, points_y])
# points = np.reshape(points, (5, 2))

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
dr = pi/15

# sail angle if you are standing on boat
# angle of stern-ward end of sail to follow RH coord system
ds = pi/3;   





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


def sumAeroVectors(lift_car, drag_car):
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


	f = lift_car + drag_car
	print("f", f)
	f_pol = cart2pol(f)
	print("fpol", f)
	return f_pol


def plot_PMW(starboard_stern_x, starboard_stern_y, centre_boat_pos):
	fig, ax = plt.subplots()
	patches = []
	boat = Rectangle((starboard_stern_x, 
					  starboard_stern_y), 
	                  boat_l, 
	                  boat_w, 
	                  angle=theta,
	                  color='c', 
	                  alpha=0.5)


	rudder_x = [pos[x] - boat_l/2, 
	            pos[x] - boat_l/2 - rudder_l]

	rudder_y = [pos[y], 
				pos[y]]

	rudder_angle = [theta, theta + dr]

	rudder_transx = []
	rudder_transy = []

	for rx, ry, a in zip(rudder_x, 
		                 rudder_y, 
		                 rudder_angle):	
		r = np.array([[rx],[ry]])

		R = [[np.cos(a), np.sin(a)],
			 [np.sin(a), np.cos(a)]]

		rudder_trans = np.dot(R, r)
		rudder_transx.append(rudder_trans[x])
		rudder_transy.append(rudder_trans[y])

	rudder = mlines.Line2D(rudder_transx, rudder_transy, linewidth=2, color='b', alpha=0.5)



	sail_x = [pos[x] + sail_l/2, 
	          pos[x] - sail_l/2]


	sail_y = [pos[y], 
			  pos[y]]

	sail_angle = [theta + ds, theta + ds]

	sail_transx = []
	sail_transy = []

	for rx, ry, a in zip(sail_x, 
		                 sail_y, 
		                 sail_angle):
		r = np.array([[rx],[ry]])

		R = [[np.cos(a), np.sin(a)],
			 [np.sin(a), np.cos(a)]]

		sail_trans = np.dot(R, r)
		sail_transx.append(sail_trans[x])
		sail_transy.append(sail_trans[y])
	
	print('sailx', sail_x)
	print('saily', sail_y)
	print('sail_angle', sail_angle)
	print('sailtx', sail_transx)
	print('sailty', sail_transy)

	sail = mlines.Line2D(sail_transx, sail_transy, linewidth=2, color='k', alpha=0.5)


	mast = Circle(pos, 0.01, color='k')


	fig1 = plt.figure()
	ax1 = fig1.add_subplot(111, aspect='equal')


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

	vectors = np.array([[pos[x], pos[y], L_s_car[x], L_s_car[y]], # sail lift
		                [pos[x], pos[y], D_s_car[x], D_s_car[y]], # sail drag
		                [pos[x], pos[y], F_s_car[x], F_s_car[y]], # sail force
		                [rudder_transx[0],rudder_transy[0], L_r_car[x], L_r_car[y]],  # rudder lift
		                [rudder_transx[0], rudder_transy[0], D_r_car[x], D_r_car[y]]]) # rudder drag

	#ax1.quiver(*origin, V[:,0], V[:,1], color=['r','b','g'], scale=21)
	

	colors = cm.rainbow(np.linspace(0, 1, len(vectors)))

	for V, c in zip(vectors, colors):
		ax1.quiver(V[0], V[1], V[2], V[3], color=c, scale=5)


	#ax1.autoscale(enable=True, axis='both', tight=None)
	ax1.set_xlim([-2, 2])
	ax1.set_ylim([-2, 2])
	plt.show()



	#plt.show()



                                   
# apparent wind
aw_car, aw_pol = appWind()

L_s_pol = aero_force(part='sail', force='lift')
D_s_pol = aero_force(part='sail', force='drag')
L_r_pol = aero_force(part='rudder', force='lift')  
D_r_pol = aero_force(part='rudder', force='drag') 

L_s_car = pol2cart(L_s_pol)
D_s_car = pol2cart(D_s_pol)
L_r_car = pol2cart(L_r_pol)
D_r_car = pol2cart(D_r_pol)

# print(f'{L_s}/n{D_s}/n{L_r}/n{D_r}')
print()
print("coordinates")
print(L_s_pol, L_s_car)
print(D_s_pol, D_s_car)
print(L_r_pol, L_r_car)
print(D_r_pol, D_r_car)

F_s_pol = sumAeroVectors(L_s_car, D_s_car)  
F_r_pol = sumAeroVectors(L_r_car, D_r_car)
print("F_s_pol", F_s_pol)
F_s_car = pol2cart(F_s_pol)

print("F_r_pol", F_r_pol)
F_r_car = pol2cart(F_r_pol)


plot_PMW(pos[x]-boat_l/2, pos[y]-boat_w/2, pos)







