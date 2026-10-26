import numpy as np


from pid import PID_ctrl
from utilities import Logger, euler_from_quaternion, calculate_angular_error, calculate_linear_error

M_PI=3.1415926535

P=0; PD=1; PI=2; PID=3

class controller:
    
    
    # Default gains of the controller for linear and angular motions
    def __init__(self, klp=0.2, klv=0.2, kli=0.2, kap=0.2, kav=0.2, kai=0.2):
        
        # TODO Part 5 and 6: Modify the below lines to test your PD, PI, and PID controller
        self.PID_linear=PID_ctrl(PID, klp, klv, kli, filename_="linear.csv")
        self.PID_angular=PID_ctrl(PID, kap, kav, kai, filename_="angular.csv")
        self.max_linear_vel = 0.22  # m/s
        self.max_angular_vel = 2.84  # rad/s
        self.velocity_logger = Logger("velocities.csv", headers=[
            "clipped", "linear_requested", "linear_command", "angular_requested",
            "angular_command", "stamp"])

    def limit_and_log(self, linear_requested, angular_requested, stamp):
        linear_vel = float(np.clip(linear_requested, -self.max_linear_vel, self.max_linear_vel))
        angular_vel = float(np.clip(angular_requested, -self.max_angular_vel, self.max_angular_vel))
        self.velocity_logger.log_values([
            int(abs(linear_requested) >= self.max_linear_vel or
                abs(angular_requested) >= self.max_angular_vel),
            linear_requested, linear_vel, angular_requested, angular_vel,
            stamp.sec * 1_000_000_000 + stamp.nanosec])
        return linear_vel, angular_vel

    
    def vel_request(self, pose, goal, status):
        
        e_lin=calculate_linear_error(pose, goal)
        e_ang=calculate_angular_error(pose, goal)


        linear_vel=self.PID_linear.update([e_lin, pose[3]], status)
        angular_vel=self.PID_angular.update([e_ang, pose[3]], status)
        
        return self.limit_and_log(linear_vel, angular_vel, pose[3])
    

class trajectoryController(controller):

    def __init__(self, klp=0.2, klv=0.2, kli=0.2, kap=0.2, kav=0.2, kai=0.2):
        
        super().__init__(klp, klv, kli, kap, kav, kai)
    
    def vel_request(self, pose, listGoals, status):
        
        goal=self.lookFarFor(pose, listGoals)
        
        finalGoal=listGoals[-1]
        
        e_lin=calculate_linear_error(pose, finalGoal)
        e_ang=calculate_angular_error(pose, goal)

        
        linear_vel=self.PID_linear.update([e_lin, pose[3]], status)
        angular_vel=self.PID_angular.update([e_ang, pose[3]], status) 

        return self.limit_and_log(linear_vel, angular_vel, pose[3])

    def lookFarFor(self, pose, listGoals):
        
        poseArray=np.array([pose[0], pose[1]]) 
        listGoalsArray=np.array(listGoals)

        distanceSquared=np.sum((listGoalsArray-poseArray)**2,
                               axis=1)
        closestIndex=np.argmin(distanceSquared)

        return listGoals[ min(closestIndex + 3, len(listGoals) - 1) ]
