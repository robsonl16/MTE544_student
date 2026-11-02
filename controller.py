import numpy as np


from pid import PID_ctrl
from utilities import Logger, euler_from_quaternion, calculate_angular_error, calculate_linear_error

M_PI=3.1415926535

P=0; PD=1; PI=2; PID=3

class controller:
    """
    Point controller: two independent PID loops, one on distance to the goal (-> linear velocity)
    and one on heading error to the goal (-> angular velocity). Outputs are saturated and logged.
    Gains: kl* = linear (p, v=derivative, i), ka* = angular.
    """

    # Default gains of the controller for linear and angular motions
    def __init__(self, controller_type=P, klp=0.2, klv=0.2, kli=0.2, kap=0.2, kav=0.2, kai=0.2):

        # Part 5 and 6: controller_type selects P / PD / PI / PID for both loops
        self.PID_linear=PID_ctrl(controller_type, klp, klv, kli, filename_="linear.csv")
        self.PID_angular=PID_ctrl(controller_type, kap, kav, kai, filename_="angular.csv")
        # Part 4: saturation limits (TurtleBot3 Burger spec values)
        self.max_linear_vel = 0.22  # m/s
        self.max_angular_vel = 2.84  # rad/s
        # Logs requested vs. clipped velocities to check when saturation is active
        self.velocity_logger = Logger("velocities.csv", headers=[
            "clipped", "linear_requested", "linear_command", "angular_requested",
            "angular_command", "stamp"])

    def limit_and_log(self, linear_requested, angular_requested, stamp):
        """Part 4: clip the PID outputs to the robot's velocity limits, log before/after, return clipped values."""
        linear_vel = float(np.clip(linear_requested, -self.max_linear_vel, self.max_linear_vel))
        angular_vel = float(np.clip(angular_requested, -self.max_angular_vel, self.max_angular_vel))
        self.velocity_logger.log_values([
            int(abs(linear_requested) >= self.max_linear_vel or
                abs(angular_requested) >= self.max_angular_vel),
            linear_requested, linear_vel, angular_requested, angular_vel,
            stamp.sec * 1_000_000_000 + stamp.nanosec])
        return linear_vel, angular_vel

    
    def vel_request(self, pose, goal, status):
        """Point mode: errors to the single goal [x, y] -> PID -> saturated (v, w)."""

        e_lin=calculate_linear_error(pose, goal)    # distance to goal [m]
        e_ang=calculate_angular_error(pose, goal)   # heading error to goal [rad]


        linear_vel=self.PID_linear.update([e_lin, pose[3]], status)
        angular_vel=self.PID_angular.update([e_ang, pose[3]], status)
        
        return self.limit_and_log(linear_vel, angular_vel, pose[3])
    

class trajectoryController(controller):
    """
    Part 6: trajectory follower. Same two PID loops, but heading is steered toward a look-ahead
    waypoint on the path while speed is based on distance to the final waypoint.
    """

    def __init__(self, controller_type=PID, klp=0.2, klv=0.2, kli=0.2, kap=0.2, kav=0.2, kai=0.2):
        
        super().__init__(controller_type, klp, klv, kli, kap, kav, kai)
    
    def vel_request(self, pose, listGoals, status):
        """Trajectory mode: heading error to the look-ahead point, linear error to the final point."""

        goal=self.lookFarFor(pose, listGoals)   # look-ahead waypoint to steer toward

        finalGoal=listGoals[-1]

        e_lin=calculate_linear_error(pose, finalGoal)   # remaining distance to end of path
        e_ang=calculate_angular_error(pose, goal)       # heading error to look-ahead point

        
        linear_vel=self.PID_linear.update([e_lin, pose[3]], status)
        angular_vel=self.PID_angular.update([e_ang, pose[3]], status) 

        return self.limit_and_log(linear_vel, angular_vel, pose[3])

    def lookFarFor(self, pose, listGoals):
        """Return the waypoint 3 ahead of the closest one to the robot (clamped to the last waypoint)."""

        poseArray=np.array([pose[0], pose[1]]) 
        listGoalsArray=np.array(listGoals)

        distanceSquared=np.sum((listGoalsArray-poseArray)**2,
                               axis=1)
        closestIndex=np.argmin(distanceSquared)

        return listGoals[ min(closestIndex + 3, len(listGoals) - 1) ]
