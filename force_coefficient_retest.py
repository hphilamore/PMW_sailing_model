import numpy as np
from numpy import deg2rad, rad2deg, pi, sin, cos, exp
import matplotlib.pyplot as plt




# Input parameters
a = 0  # angle of attack, degrees
AR = 5 # Aspect ratio
c = 0.2 # chord, m
t = 0 # thickness




def aero_coeffs(attack_angle, aspect_ratio, chord, thickness, CN1max_infinite, CDmin_infinite):
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
	CD0 = CDmin_infinite #0
	ACL1_inf = 9 #degrees
	ACD1_inf = ACL1_inf
	CN1max_inf = CN1max_infinite
	CL1max_inf = cos(deg2rad(ACL1_inf)) * CN1max_infinite
	CD1max_inf = sin(deg2rad(ACL1_inf)) * CN1max_infinite
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

	return CL1, CD1, CL, CL2, CD2, CD

attack_angle = np.linspace(0, pi, 100)

# aero_coeffs_vec = np.vectorize(aero_coeffs)

# # #(attack_angle, aspect_ratio, chord, thickness, CN1max_infinite, CDmin_infinite)

# CL1, CD1, CL, CL2, CD2, CD, CL1max_inf, CD1max_inf, CL1max, CD1max = aero_coeffs_vec(
# 	attack_angle, AR, c, t, 1, 0)
# cl = CL
# cd = CD
# cl1 = CL1
# cd1 = CD1
# cl2 = CL2
# cd2 = CD2



cl1 = []
cl2 = []
cl = []

cd1 = []
cd2 = []
cd = []

# boat parameters
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
CN1max_inf_h = 1


# hull minimum drag coefficient
CD0_hull = 0.1

c_s= sail_l  #0.2 # chord, m
t_s = 0 # thickness
c_r= rudder_l# 0.2 # chord, m
t_r = 0 # thickness
c_h= boat_l # 0.2 # chord, m
t_h = boat_l/2 # thickness


# # Input parameters
# a = 0  # angle of attack, degrees
# AR = 5 # Aspect ratio
# c = 0.2 # chord, m
# t = 0 # thickness
# CN1max_inf = 1
# CDmin_inf = 0

# # Input parameters
# a = 0  # angle of attack, degrees
# AR = 4 # Aspect ratio
# c = 0.2 # chord, m
# t = 0 # thickness
# CN1max_inf = 1
# CDmin_inf = 0


# # #a = 0  # angle of attack, degrees
# # AR = ARs#5 # Aspect ratio
# # c = c_s#0.2 # chord, m
# # t = t_s #0 # thickness
# # CN1max_inf = CN1max_inf_s
# # CDmin_inf = 0


AR = ARr#5 # Aspect ratio
c = c_r#0.2 # chord, m
t = t_r #0 # thickness
CN1max_inf = CN1max_inf_r
CDmin_inf = 0

# AR = ARh#5 # Aspect ratio
# c = c_h#0.2 # chord, m
# t = t_h #0 # thickness
# CN1max_inf = CN1max_inf_h
# CDmin_inf = CD0_hull


# AR = 2 # Aspect ratio
# c = 0.2 # chord, m
# t = 0 # thickness
# CN1max_inf = 1#0.75#CN1max_inf_s
# CDmin_inf = 0

for a in attack_angle:
	CL1, CD1, CL, CL2, CD2, CD = aero_coeffs(a, AR, c, t, CN1max_inf, CDmin_inf)
	cl.append(CL)
	cd.append(CD)
	cl1.append(CL1)
	cd1.append(CD1)
	cl2.append(CL2)
	cd2.append(CD2)

attack_angle = rad2deg(attack_angle)


# print('CL1max_inf= ' , CL1max_inf)
# print('CD1max_inf= ' , CD1max_inf)
# print('CL1max= ' , CL1max)
# print('CD1max= ' , CD1max)

plt.plot(attack_angle, cl, label='lift')
plt.plot(attack_angle, cd, label='drag')
plt.plot(attack_angle, cl1, label='lift1')
plt.plot(attack_angle, cd1, label='drag1')
plt.plot(attack_angle, cl2, label='lift2')
plt.plot(attack_angle, cd2, label='drag2')
plt.legend()

plt.xlim((0, 90))
plt.ylim((0, 2))
plt.show()




