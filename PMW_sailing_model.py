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



fig1, ax1 = plt.subplots()


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
tw_pol = np.array([pi, 3])
aw_pol = appWind(tw_pol, v_pol)
aw_car = pol2cart(aw_pol)
# rudder angle if you are standing on boat
# % -ve towards port (left)
# % +ve towards starboard (right)
ra = pi/15

# sail angle if you are standing on boat
# angle of stern-ward end of sail to follow RH coord system
sa = pi/3; 
#Z_init_state = [pos_car[x], pos_car[y], theta, v_pol[1] , w]
Z_init_state = [pos_pol, 
				v_pol,
				theta,
				w] 






  




def attack_angle(part_angle, area, v_fluid_pol):
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

		# convert angles to cartesian
		plane_car = pol2cart([part_angle, 1])
		v_fluid_car = pol2cart(v_fluid_pol)

		# use dot product to find angle cosine
		U = plane_car
		V = v_fluid_car
		cosalpha = np.dot(U, V) / np.dot(np.linalg.norm(U), np.linalg.norm(V))
		print("cosalpha", cosalpha)

		alpha = abs(np.arccos(cosalpha))

		# find smallest of two possible angles
		if alpha>pi/2:
			alpha = 180 - alpha

		print("alpha", alpha)

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
	if fab > 2*pi:
		fab -= 2*pi
	elif fab < 0:
		fab += 2*pi 

	#v_fluid_boat_pol = pol2cart([vfab, v_fluid_pol[1]])


	#establish orientation or sail or rudder  

	if (pa_car[x] / pa_car[y]) < 0:	# 2nd or 4th quadrant 

	    
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

def aero_force(part, force, v_fluid_pol, part_angle):
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
	alpha = attack_angle(part_angle, area=A, v_fluid_pol=v_fluid_pol)

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

	# aero force
	if force == 'lift':
		lift_a = lift_angle(part_angle, area=A, v_fluid_pol=v_fluid_pol)
		lift_force = 0.5 * rho * A * v_fluid_pol[1]**2 * CL
		return np.array([lift_a, lift_force])

	else: # force = 'drag'
		drag_angle = v_fluid_pol[0];
		drag_force = 0.5 * rho * A * v_fluid_pol[1]**2 * CD
		return np.array([drag_angle, drag_force])


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
	
	print('sailx', sail_x)
	print('saily', sail_y)
	print('sail_angle', sail_angle)
	print('sailtx', sail_transx)
	print('sailty', sail_transy)

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

	vectors = np.array([[pos[x], pos[y], L_s_car[x], L_s_car[y]], # sail lift
		                [pos[x], pos[y], D_s_car[x], D_s_car[y]], # sail drag
		                [pos[x], pos[y], F_s_car[x], F_s_car[y]], # sail force
		                [rudder_transx[0],rudder_transy[0], L_r_car[x], L_r_car[y]],  # rudder lift
		                [rudder_transx[0], rudder_transy[0], D_r_car[x], D_r_car[y]]]) # rudder drag
	labels = ['Lsail', 'Dsail', 'Fsail', 'Lrud', 'Drud', 'Frud']
	#ax1.quiver(*origin, V[:,0], V[:,1], color=['r','b','g'], scale=21)
	# QV1 = plt.quiver(x, y, u1, v1, color='r')
	# plt.quiverkey(QV1, 1.2, 0.515, 2, 'arrow 1', coordinates='data')
	

	colors = cm.rainbow(np.linspace(0, 1, len(vectors)))

	for n, (V, c, label) in enumerate(zip(vectors, colors, labels), 1):
		# ax1.quiver(V[0], V[1], V[2], V[3], color=c, scale=5)
		Q = plt.quiver(V[0], V[1], V[2], V[3], color=c, scale=5)
		plt.quiverkey(Q, -1.5, n/2-2, 0.25, label, coordinates='data')


	#ax1.autoscale(enable=True, axis='both', tight=None)
	ax1.set_xlim([-2, 2])
	ax1.set_ylim([-2, 2])
	#plt.show()

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
	# # angle of force relative to boat
	# Fsba = Fs_pol[0] - theta 
	# Frba = Fs_pol[0] - theta 

	# # component of sail force in direction of boat heading
	# Fsb_fwd_pol = np.array([Fsba, 
	# 	                Fs_pol[1]*np.cos(Fsba)])

	# # component of sail force acting perpendicular to boat heading
	# Fsb_side_pol = np.array([Fsba, 
	# 	                Fs_pol[1]*np.sin(Fsba)])

	# # component of sail force in direction of boat heading
	# Frb_fwd_pol = np.array([Frba,
	# 	                Fr_pol[1] * np.cos(Frba)])

	# # component of sail force acting perpendicular to boat heading
	# Frb_side_pol = np.array([Frba,
	# 	                Fr_pol[1] * np.sin(Frba)])
	print('F_s_pol2', Fs_pol)
	Fsb_fwd_pol, Fsb_side_pol, Frb_fwd_pol, Frb_side_pol = force_on_boat(Fs_pol, Fr_pol, theta)


	Fsb_fwd_car = pol2cart(Fsb_fwd_pol)
	Fsb_side_car = pol2cart(Fsb_side_pol)
	Frb_fwd_car = pol2cart(Frb_fwd_pol)
	Frb_side_car = pol2cart(Frb_side_pol)


	Fb_fwd_car = Fsb_fwd_car + Frb_fwd_car          # sail + rudder force, direction of heading, boat frame of ref
	Fb_fwd_pol = cart2pol(Fb_fwd_car)				# cartesian coordinates
	F_fwd_pol = Fb_fwd_pol + np.array([theta, 0])   # global coords
	F_fwd_car = cart2pol(F_fwd_pol)					# global cartesian coords	


	Fb_side_car = f_leering(Fsb_side_car)
	Fb_side_pol = cart2pol(Fb_fwd_car)		     	 # cartesian coordinates
	F_side_pol = Fb_fwd_pol + np.array([theta, 0])   # global coords
	F_side_car = cart2pol(F_side_pol)             # global cartesian coords   

	F_car = Fb_fwd_car + F_side_car

	mass = 1

	a = F_car / mass

	#F = np.array(F_fwd_car, F_side_car)

	return cart2pol(a)

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

	print("v_pol", v_pol)
	v_car = pol2cart(v_pol)

	dvdt_car = pol2cart(dvdt(Fs_pol, Fr_pol, theta))

	return cart2pol(v_car + dvdt_car) 
		            


def force_on_boat(Fs_pol, Fr_pol, theta):

	"""
	Resolves the sail and rudder force perpendiular and parallel to the boat (bow to stern) axis"
	"""

	# angle of force relative to boat
	print('Fs_pol1', Fs_pol)
	Fsba = Fs_pol[0] - theta 
	Frba = Fs_pol[0] - theta 

	# component of sail force in direction of boat heading
	Fsb_fwd_pol = np.array([Fsba, 
		                Fs_pol[1]*np.cos(Fsba)])

	# component of sail force acting perpendicular to boat heading
	Fsb_side_pol = np.array([Fsba, 
		                Fs_pol[1]*np.sin(Fsba)])

	# component of sail force in direction of boat heading
	Frb_fwd_pol = np.array([Frba,
		                Fr_pol[1] * np.cos(Frba)])

	# component of sail force acting perpendicular to boat heading
	Frb_side_pol = np.array([Frba,
		                Fr_pol[1] * np.sin(Frba)])

	return Fsb_fwd_pol, Fsb_side_pol, Frb_fwd_pol, Frb_side_pol


def dwdt(w, Fs_pol, Fr_pol, theta, rudder_a):
	"""
	The angular velocity of the boat due to the rudder moment
	"""

	print('Fs_pol3', Fs_pol)
	fFsb_fwd_pol, Fsb_side_pol, Frb_fwd_pol, Frb_side_pol = force_on_boat(Fs_pol, Fr_pol, theta)

	return (Frb_side_pol[1] * 
		   (boat_l/2 + abs(rudder_l*np.cos(rudder_a))) + # moment due to force on rudder
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
	pos_pol = Z_state[0]
	v_pol =   Z_state[1]
	theta =   Z_state[2]
	w =       Z_state[3]

	data["sail_angle"].append(sa)
	data["rudder_angle"].append(ra)
	data["sail_area"].append(A_s)

	aw_pol = appWind(tw_pol, v_pol)
	
	# calculate lift and drag force
	Ls_pol = aero_force(part='sail',   force='lift', v_fluid_pol=aw_pol, part_angle=sa)
	Ds_pol = aero_force(part='sail',   force='drag', v_fluid_pol=aw_pol, part_angle=sa)
	Lr_pol = aero_force(part='rudder', force='lift', v_fluid_pol=v_pol, part_angle=ra)  
	Dr_pol = aero_force(part='rudder', force='drag', v_fluid_pol=v_pol, part_angle=ra) 

	Fs_pol = sumAeroVectors(Ls_pol, Ds_pol)  
	Fr_pol = sumAeroVectors(Lr_pol, Dr_pol)

	print("Fs_pol", Fs_pol)
	print("Fr_pol", Fr_pol)

	dZdt = [dpdt(v_pol, Fs_pol, Fr_pol, theta), 
	        dvdt(Fs_pol, Fr_pol, theta), 
	        dthdt(w, Fs_pol, Fr_pol, theta, ra),  
	        dwdt(w, Fs_pol, Fr_pol, theta, ra),
			]

	return np.array(dZdt)
  


# main program
time = np.arange(0, 20, 1)

#sail_angle, rudder_angle, sail_area, position, velocity, heading, angular_vel = [], [], [], [], [], [], []
data = {'sail_angle' : [], 'rudder_angle' : [], 'sail_area' : [], 'position' : [], 'velocity' : [], 'heading' : [], 'angular_vel' : []}


for t in time:


	state = param_solve(Z_init_state)

	for i in range(len(Z_init_state)):
		Z_init_state[i] = Z_init_state[i] + state[i]
		
	data["position"].append(Z_init_state[0])
	data["velocity"].append(Z_init_state[1])
	data["heading"].append(Z_init_state[2])	
	data["angular_vel"].append(Z_init_state[3])

#state = odeint(param_solve, Z_init_state, time)
print(data["sail_angle"])

fig2, ax2 = plt.subplots()

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
plt.show()

	







