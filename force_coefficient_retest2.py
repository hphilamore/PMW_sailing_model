def aero_coeffs(attack_angle, aspect_ratio, chord, thickness, CN1max_infinite, CDmin_infinite):
	"""
	Computes the lift abd drag coefficient for a given angle of attack.
	Considers pre and post stall condition up to 90 degrees

	"""
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

	return CL, CD

attack_angle = np.linspace(0, pi, 100)