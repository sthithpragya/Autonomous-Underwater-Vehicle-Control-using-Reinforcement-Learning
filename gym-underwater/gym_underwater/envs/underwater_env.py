# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:26:08 2019

@author: Sthithpragya Gupta
"""
# ADD full 6 dimensional position and velocity vectors
import csv
import gym
from gym import spaces
import random
import math
import numpy as np
from numpy import *
# from helperFunc import *

class UnderwaterEnv(gym.Env):
	"""Custom Environment that follows gym interface"""
	metadata = {'render.modes': ['human']}
	def __init__(self):
		# defining the span of action space
		self.lowActuation		= np.asarray([fMin]*thrusterCount)
		self.highActuation 		= np.asarray([fMax]*thrusterCount)
		self.action_space 		= spaces.Box(self.lowActuation,self.highActuation,dtype=np.float32)

		# defining the span of observation space
		# current obs space/ state = [goal disp, current velocity]
		# other states to try: [goal disp, current velocity, prev actuation, ocean forces]
		# NEXT: try adding prev actutation to current state

		self.lowObservation		= np.asarray([xMin]*inputDispDimLin + [math.pi]*inputDispDimAng  + [xdotMin]*currentVelDimLin + [math.pi]*currentVelDimLin)# + inputVelDim))# + [fsensorMin]*oceanInfoDim)
		self.highObservation	= np.asarray([xMax]*inputDispDimLin + [-math.pi]*inputDispDimAng + [xdotMax]*currentVelDimLin + [-math.pi]*currentVelDimLin)# + inputVelDim))# + [fsensorMax]*oceanInfoDim)
		self.observation_space	= spaces.Box(self.lowObservation, self.highObservation, dtype=np.float32)

	def step(self, action):

		# timeElap 		= self.timeElapsed  # track of time elapsed
		currentState	= self.state

		goalDisp 		= currentState[0:3] # x, y, z
		# print("goal 		", goalDisp)
		currentVel 		= currentState[6:12]
		# goalVel			= currentState[3:6]
		# currentVel 		= currentState[6:9]
		# currentEul 		=
		dynamics = Dynamics()

		dynamics.v_0[0] = dynamics.v[0] = currentVel[0]
		dynamics.v_0[1] = dynamics.v[1] = currentVel[1]
		dynamics.v_0[2] = dynamics.v[2] = currentVel[2]
		dynamics.v_0[3] = dynamics.v[3] = currentVel[3]
		dynamics.v_0[4] = dynamics.v[4] = currentVel[4]
		dynamics.v_0[5] = dynamics.v[5] = currentVel[5]

		# current thruster actuations
		dynamics.u      = action


		#MAKE THRUSTER FORCE AN ATTRIBUTE AND USE IT IN CLASS FUNCTIONS
		# DO DYNAMICS.THRUSTERFORCE = ACTION

		# calculate the new postion and velocities under ocean effects
		dynamics.iterate()

#        newVel          = dynamics.v[0:3]
		realDisp		= dynamics.p[0:3]
		# print("real disp 	", realDisp)
		rollPitchYaw 	= dynamics.p[3:6]

		# overshoot to evaluate reward
		overshoot       = dist(goalDisp,realDisp)

		done = bool(overshoot >= xMax*1.73)

		reward = 0 # reward initialisation
		# reward allocation scheme wrt article

		if(overshoot <= beta):
			reward = 10 # ideal case
		elif(overshoot > beta and overshoot < betaMax):
			reward = -1 - math.floor((overshoot - beta)/betaStep) # relatively bad
		else:
			reward = -10 # worst

		#new state
		newGoalDisp     = (goalDisp - realDisp).tolist()
		# print("newGoalDisp 	", newGoalDisp)
		rollPitchYaw 	= dynamics.p[3:6].tolist()
		newCurrentVel   = dynamics.v[0:6].tolist()

		self.state 		= np.asarray(newGoalDisp + rollPitchYaw + newCurrentVel)
		# print("goal ", goalDisp)
		# print(realDisp)
		# print(newGoalDisp)

		return self.state, reward, done, {}

	def reset(self):
		# random set goal disp
		# tempDisp        = np.random.uniform(xMin,xMax,(6,))
		# tempDisp        = np.round(tempDisp, 2) # least count is 1 mm
		tempDispLin 	= np.random.uniform(xMin,xMax,(inputDispDimLin,))
		tempDispAng 	= np.random.uniform(-math.pi,math.pi,(inputDispDimAng,))
		tempDisp 		= np.append(tempDispLin, tempDispAng)
		tempDisp        = np.round(tempDisp, 2) # least count is 1 mm

		tempDisp[3] 	= 0 # setting roll to be 0 since ideally it can't be altered


		
		tempVelLin 		= np.random.uniform(xdotMin,xdotMax,(currentVelDimLin,))
		tempVelAng		= np.random.uniform(-math.pi,math.pi,(currentVelDimAng,))
		tempVel 		= np.append(tempVelLin, tempVelAng)
		tempVel         = np.round(tempVel, 2) # least count is 1 mm

		tempVel[3]		= 0

		tempState       = np.append(tempDisp, tempVel)
		# doing the update
		self.state      = tempState
		print("made a reset")
		return np.array(self.state)

	# def setGoal(self, goalDataList):
	# 	goalState 		= np.array(goalDataList)
	# 	self.state 		= goalState
	# 	return np.array(self.state)

	def render(self, mode='human', close=False):
		# printing the current stats and also writing to a CSV file
		# print('state is:  ',self.state, '  time elapsed: ',self.timeElapsed)
		with open('trainData.csv', 'a', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(self.state)


############################ ROBOT PARAM #################################

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:26:46 2019

@author: Sthithpragya Gupta
"""
#Robot specs
thrusterCount	= 5
fMax			= 1 	# N
fMin			= -1	# N

#Trajectory specs
dT				= 0.01		# sec from the article
#unsure as dynamic model uses 0.001 and article recommends 0.1

#User info specs
inputDispDimLin	= 3		# displacement in x, y, z, roll, pitch, yaw - though goal is not concerned with pitch and yaw
inputDispDimAng	= 3		# displacement in x, y, z, roll, pitch, yaw - though goal is not concerned with pitch and yaw
#inputVelDim		= 3		# goal xdot, ydot, zdot
currentVelDimLin= 3		# current xdot, ydot, zdot, rolldot, pitchdot, yawdot
currentVelDimAng= 3		# current xdot, ydot, zdot, rolldot, pitchdot, yawdot

#oceanInfoDim	= 3		# 3 force sensors

xMax			= 250    # max displacement in x (cm)
xMin			= -250   # min displacement in x (cm)
# yawMax 			= math.pi
# yawMin 			= -math.pi

xdotMax			= 500	# cm/s max xdot vehicle can achieve (limit unsure, moreover not really concerned)
xdotMin			= -500	# cm/s min xdot vehicle can achieve (limit unsure, moreover not really concerned)
# yawdotMax		= 1
# yawdotMin 		= -1

fsenorMax		= 1		# max force sensed during oceanic disturbances
fsenorMin		= -1	# min force sensed during oceanic disturbances

# beta to be decayed as episode progress
# intial estimate is based from the article

###betaPoor
beta            = 40 #cm
betaStep		= 13 #cm
betaMax         = 175 #cm

###betaInter
# beta            = 25 #cm
# betaStep		= 12.5 #cm
# betaMax         = 150 #cm

###betaGood
# beta            = 15 #cm
# betaStep		= 11 #cm
# betaMax         = 125 #cm

###betaBetter
#beta            = 5 #cm
#betaStep		= 7.5 #cm
#betaMax         = 80 #cm

###betaBest
#beta            = 5 #cm
#betaStep		= 5 #cm
#betaMax         = 55 #cm



#Oceanic currents data - can be altered in later episodes once beta has been reduced to the least
currentMean		= [0.0, 0.0, 0.0]
currentSigma 	= [0.0, 0.0, 0.0]
currentMin 		= [0.0, 0.0, 0.0]
currentMax 		= [0.0, 0.0, 0.0]


cm2m            = 0.01 # converting cm to m
xMax            = xMax*cm2m
xMin            = xMin*cm2m
xdotMax         = xdotMax*cm2m
xdotMin         = xdotMin*cm2m
beta            = beta*cm2m
betaStep		= betaStep*cm2m
betaMax         = betaMax*cm2m





def dist(goal,real):
    xtemp = goal[0] - real[0]
    ytemp = goal[1] - real[1]
    ztemp = goal[2] - real[2]
    temp = math.pow(xtemp,2) + math.pow(ytemp,2) + math.pow(ztemp,2)
    return math.sqrt(temp)

######################################### HELPER #########################

class Dynamics :

	def getConfig(self) :
		""" Load parameters from the rosparam server """
		self.num_actuators = thrusterCount
		self.period = dT
		self.mass = 98.0
		self.gravity_center = [0.0, 0.0, 0.05]
		self.g = 9.81
		self.radius = 0.286
		self.ctf = 0.00006835
		self.ctb = 0.00006835
		self.actuators_tau = [0.2, 0.2, 0.2, 0.2, 0.2]
		self.actuators_maxsat= [1, 1, 1, 1, 1]
		self.actuators_minsat = [-1, -1, -1, -1, -1]
		self.actuators_gain = [1500, 1500, 1500, 1500, 1500]
		self.dzv = 0.05
		self.dv = 0.35
		self.dh = 0.4
		self.density = 1000.0
		self.tensor = [8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0]
		self.damping = [.0, .0, .0, -130.0, -130.0, -130.0]
		self.quadratic_damping = [-148.0, -148.0, -148.0, -180.0, -180.0, -180.0]

		self.p_0 = [0.0, 0.0, 0.0, 0, 0, 1.57] #[3.0, 1.1, 2.8, 0, 0, 3.14]
		self.v_0 = [0, 0, 0, 0, 0, 0]
		#self.frame_id = rospy.get_param(self.vehicle_name + "/dynamics" + "/frame_id")
		# self.external_force_topic = rospy.get_param(self.vehicle_name + "/dynamics" + "/external_force_topic")

# 		self.am = [-ct[0]*abs(du[0]),             -ct[1]*abs(du[1]),              .0,                       .0,                           .0,
# .0,                             .0,                             .0,                       .0,                           ct[4]*abs(du[4]),
# .0,                             .0,                             -ct[2]*abs(du[2]),          -ct[3]*abs(du[3]),            .0,
# .0,                             .0,                             .0,                       .0,                           .0,
# .0,                             .0,                             -ct[2]*self.dv*abs(du[2]), ct[3]*self.dv*abs(du[3]),     .0,
# -ct[0]*self.dh*abs(du[0]),      ct[1]*self.dh*abs(du[1]),       .0,                       .0,                           .0]

#       Currents data
		self.current_mean = currentMean
		self.current_sigma = currentSigma
		self.current_min = currentMin
		self.current_max = currentMax

		self.uwsim_period = dT


	def s(self, x) :
		""" Given a 3D vector computes the 3x3 antisymetric matrix """
#        rospy.loginfo("s(): \n %s", x)
		ret = array([0.0, -x[2], x[1], x[2], 0.0, -x[0], -x[1], x[0], 0.0 ])
		return ret.reshape(3,3)


	def generalizedForce(self, du):
		# Computes the generalized force as B*u, being B the allocation matrix and u the control input
		ct = zeros(len(du))
		i1 = nonzero(du >= 0.0)
		i2 = nonzero(du <= 0.0)
		ct[i1] = self.ctf
		ct[i2] = self.ctb

		#Evaluates allocation matrix loaded as parameter
		# ADDED self.am AS A LOCAL VARIABLE HERE
		am = [-ct[0]*abs(du[0]),             -ct[1]*abs(du[1]),              .0,                       .0,                           .0,
.0,                             .0,                             .0,                       .0,                           ct[4]*abs(du[4]),
.0,                             .0,                             -ct[2]*abs(du[2]),          -ct[3]*abs(du[3]),            .0,
.0,                             .0,                             .0,                       .0,                           .0,
.0,                             .0,                             -ct[2]*self.dv*abs(du[2]), ct[3]*self.dv*abs(du[3]),     .0,
-ct[0]*self.dh*abs(du[0]),      ct[1]*self.dh*abs(du[1]),       .0,                       .0,                           .0]


		# b=eval(am)
		# b=array(b).reshape(6,size(b)/6)
		b=array(am).reshape(6,int(size(am)/6))

		# t = generalized force
		t = dot(b, du)
		t = squeeze(asarray(t)) #Transforms a matrix into an array
		return t


	def coriolisMatrix(self):
		s1 = self.s(dot(self.M[0:3,0:3], self.v[0:3]) + dot(self.M[0:3,3:6], self.v[3:6]))
		s2 = self.s(dot(self.M[3:6,0:3], self.v[0:3]) + dot(self.M[3:6,3:6], self.v[3:6]))
		c = zeros((6, 6))
		c[0:3,3:6] = -s1
		c[3:6,0:3] = -s1
		c[3:6,3:6] = -s2
		return c

	def dumpingMatrix(self):
		# lineal hydrodynamic damping coeficients
		Xu = self.damping[0]
		Yv = self.damping[1]
		Zw = self.damping[2]
		Kp = self.damping[3]
		Mq = self.damping[4]
		Nr = self.damping[5]

		# quadratic hydrodynamic damping coeficients
		Xuu = self.quadratic_damping[0]    #[Kg/m]
		Yvv = self.quadratic_damping[1]    #[Kg/m]
		Zww = self.quadratic_damping[2]    #[Kg/m]
		Kpp = self.quadratic_damping[3]    #[Kg*m*m]
		Mqq = self.quadratic_damping[4]    #[Kg*m*m]
		Nrr = self.quadratic_damping[5]    #[Kg*m*m]

		d = diag([Xu + Xuu*abs(self.v[0]),
				  Yv + Yvv*abs(self.v[1]),
				  Zw + Zww*abs(self.v[2]),
				  Kp + Kpp*abs(self.v[3]),
				  Mq + Mqq*abs(self.v[4]),
				  Nr + Nrr*abs(self.v[5])])
		return d

	def gravity(self):
		# Computes the gravity and buoyancy forces. Assumes a sphere model for now
		#Weight and Flotability
		W = self.mass * self.g # [Kg]

		#If the vehicle moves out of the water the flotability decreases
		#FIXME: Assumes water surface at 0.0. Get this value from uwsim.
		if self.p[2] < 0.0:
			r = self.radius + self.p[2]
			if r < 0.0:
				r = 0.0
		else :
			r = self.radius

		#TODO: either set as parameter, since different functions may be desired for different vehicles
		# or define common models and let the user choose one by the name
		# Eventually let this part to bullet inside uwsim (HfFluid)
		F = ((4 * math.pi * pow(r,3))/3)*self.density*self.g

		# gravity center position in the robot fixed frame (x',y',z') [m]
		zg = self.gravity_center[2]

		g = array([(W - F) * sin(self.p[4]),
				   -(W - F) * cos(self.p[4]) * sin(self.p[3]),
				   -(W - F) * cos(self.p[4]) * cos(self.p[3]),
				   zg*W*cos(self.p[4])*sin(self.p[3]),
				   zg*W*sin(self.p[4]),
				   0.0])

		return g


	def inverseDynamic(self) :
		""" Given the setpoint for each thruster, the previous velocity and the
			previous position computes the v_dot """
		du = self.thrustersDynamics(self.u)
		t = self.generalizedForce(du)
		c = self.coriolisMatrix()
		d = self.dumpingMatrix()
		g = self.gravity()
		c_v = dot((c-d), self.v)
		v_dot = dot(self.IM, (t-c_v-g+self.collisionForce)) #t-c_v-g+collisionForce
		v_dot = squeeze(asarray(v_dot)) #Transforms a matrix into an array
		self.collisionForce=[0,0,0,0,0,0]
		return v_dot

	def integral(self, x_dot, x, t) :
		""" Computes the integral o x dt """
		return (x_dot * t) + x


	def kinematics(self) :
		""" Given the current velocity and the previous position computes the p_dot """
		roll = self.p[3]
		pitch = self.p[4]
		yaw = self.p[5]

		rec = [cos(yaw)*cos(pitch), -sin(yaw)*cos(roll)+cos(yaw)*sin(pitch)*sin(roll), sin(yaw)*sin(roll)+cos(yaw)*cos(roll)*sin(pitch),
			   sin(yaw)*cos(pitch), cos(yaw)*cos(roll)+sin(roll)*sin(pitch)*sin(yaw), -cos(yaw)*sin(roll)+sin(pitch)*sin(yaw)*cos(roll),
			   -sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)]
		rec = array(rec).reshape(3,3)

		to = [1.0, sin(roll)*tan(pitch), cos(roll)*tan(pitch),
			  0.0, cos(roll), -sin(roll),
			  0.0, sin(roll)/cos(pitch), cos(roll)/cos(pitch)]

		to = array(to).reshape(3,3)

		p_dot = zeros(6)
		p_dot[0:3] = dot(rec, self.v[0:3])
		p_dot[3:6] = dot(to, self.v[3:6])
		return p_dot

	def updateThrusters(self, thrusters) :
		"""Receives the control input, saturates each component to maxsat or minsat, and multiplies each component by the actuator gain"""
		#TODO: Check the size of thrusters.data
		t = array(thrusters.data)
		for i in range(size(t)):
			if t[i]>self.actuators_maxsat[i]:
				t[i]=self.actuators_maxsat[i]
			elif t[i]<self.actuators_minsat[i]:
				t[i]=self.actuators_minsat[i]
		self.u=t
		for i in range(size(t)):
			self.u[i] = self.u[i]*self.actuators_gain[i]

	def thrustersDynamics(self, u):
		y = zeros(size(u))
		for i in range(size(u)):
			y[i] = (self.period * u[i] + self.actuators_tau[i] * self.y_1[i]) / (self.period + self.actuators_tau[i])

		self.y_1 = y
		return y

	def updateCollision(self, force):
		self.collisionForce=[force.wrench.force.x,force.wrench.force.y,force.wrench.force.z,force.wrench.torque.x,force.wrench.torque.y,force.wrench.torque.z]

	# HERE DO STUFF
	# def pubPose(self, event):
	# 	pose = Pose()

	# 	pose.position.x = self.p[0]
	# 	pose.position.y = self.p[1]
	# 	pose.position.z = self.p[2]

	# 	orientation = tf.transformations.quaternion_from_euler(self.p[3], self.p[4], self.p[5], 'sxyz')
	# 	pose.orientation.x = orientation[0]
	# 	pose.orientation.y = orientation[1]
	# 	pose.orientation.z = orientation[2]
	# 	pose.orientation.w = orientation[3]

	# 	self.pub_pose.publish(pose)

	# 	# Broadcast transform
	# 	br = tf.TransformBroadcaster()
	# 	br.sendTransform((self.p[0], self.p[1], self.p[2]), orientation,
	# 	rospy.Time.now(), "world", str(self.frame_id))

	def computeTf(self, tf):
		r = PyKDL.Rotation.RPY(math.radians(tf[3]), math.radians(tf[4]), math.radians(tf[5]))
		v = PyKDL.Vector(tf[0], tf[1], tf[2])
		frame = PyKDL.Frame(r, v)
		return frame

	def reset(self,req):
		self.v = self.v_0
		self.p = self.p_0
		return []

	def __init__(self):
		""" Simulates the dynamics of an AUV """

		# if len(sys.argv) != 6:
		#   sys.exit("Usage: "+sys.argv[0]+" <namespace> <input_topic> <output_topic>")

		# self.namespace=sys.argv[1]
		# self.vehicle_name=self.namespace
		# self.input_topic=sys.argv[2]
		# self.output_topic=sys.argv[3]

		#  Collision parameters
		self.collisionForce = [0,0,0,0,0,0]

	#   Load dynamic parameters
		self.getConfig()
		#self.altitude = -1.0
		self.y_1 = np.zeros(5)

	#   Create publisher
		# self.pub_pose= rospy.Publisher(self.output_topic, Pose)
		# rospy.init_node("dynamics_"+self.vehicle_name)

	#   Init pose and velocity and period
		self.v = self.v_0
		self.p = self.p_0

		# Inertia Tensor. Principal moments of inertia, and products of inertia [kg*m*m]
		Ixx = self.tensor[0]
		Ixy = self.tensor[1]
		Ixz = self.tensor[2]
		Iyx = self.tensor[3]
		Iyy = self.tensor[4]
		Iyz = self.tensor[5]
		Izx = self.tensor[6]
		Izy = self.tensor[7]
		Izz = self.tensor[8]
		m = self.mass
		xg = self.gravity_center[0]
		yg = self.gravity_center[1]
		zg = self.gravity_center[2]

		Mrb = [ 98.0,    0.0,    0.0,    0.0,    4.9,  -0.0,
					 0.0,   98.0,    0.0,   -4.9,   0.0,    0.0,
					 0.0,    0.0,   98.0,    0.0,   -0.0,    0.0,
					 0.0,   -4.9,   0.0,    8.0,    0.0,    0.0,
					 4.9,   0.0,   -0.0,    0.0,    8.0,    0.0,
					 -0.0,    0.0,    0.0,    0.0,    0.0,    8.0 ]
		Mrb = array(Mrb).reshape(6, 6)

		# Inertia matrix of the rigid body
		# Added Mass derivative
		Ma =[  49.0,    0.0,    0.0,    0.0,    0.0,   0.0,
					 0.0,    49.0,    0.0,    0.0,    0.0,   0.0,
					 0.0,    0.0,    49.0,    0.0,    0.0,   0.0,
					 0.0,    0.0,    0.0,    0.0,    0.0,   0.0,
					 0.0,    0.0,    0.0,    0.0,    0.0,   0.0,
					 0.0,    0.0,    0.0,    0.0,    0.0,   0.0 ]
		Ma = array(Ma).reshape(6, 6)

		self.M = Mrb + Ma    # mass matrix: Mrb + Ma
		self.IM = matrix(self.M).I
#        rospy.loginfo("Inverse Mass Matrix: \n%s", str(self.IM))

		#Init currents
		random.seed()
		self.e_vc = self.current_mean
	#The number of zeros will depend on the number of actuators
		self.u = array(zeros(self.num_actuators)) # Initial thrusters setpoint

		#Publish pose to UWSim
		# rospy.Timer(rospy.Duration(self.uwsim_period), self.pubPose)

	#   Create Subscribers for thrusters and collisions
	#TODO: set the topic names as parameters
		# rospy.Subscriber(self.input_topic, Float64MultiArray, self.updateThrusters)
		# rospy.Subscriber(self.external_force_topic, WrenchStamped, self.updateCollision)


		# s = rospy.Service('/dynamics/reset',Empty, self.reset)

	def iterate(self):
		# t1 = rospy.Time.now()

		# Main loop operations
		self.v_dot = self.inverseDynamic()
		self.v = self.integral(self.v_dot, self.v, self.period)
		self.p_dot = self.kinematics()
		self.p = self.integral(self.p_dot, self.p, self.period)

		# t2 = rospy.Time.now()
		# p = self.period - (t2-t1).to_sec()
		# if p < 0.0 : p = 0.0
		# rospy.sleep(p)