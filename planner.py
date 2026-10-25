import numpy as np
# Type of planner
POINT_PLANNER=0; TRAJECTORY_PLANNER=1


class planner:
    def __init__(self, type_):

        self.type=type_
        self.goal_point=None
    
    def plan(self, goalPoint=[-1.0, -1.0]):
        self.goal_point=goalPoint

        if self.type==POINT_PLANNER:
            return self.point_planner(0)
        
        elif self.type==TRAJECTORY_PLANNER:
            return self.trajectory_planner(1)


    def point_planner(self, goalPoint):
        x = goalPoint[0]
        y = goalPoint[1]
        return x, y

    # TODO Part 6: Implement the trajectories here
    def trajectory_planner(self, trajectory_type=0, ):
        if trajectory_type == 0:
            # TODO Part 6: Implement parabola
            # goal_x = np.clip(self.goal_point[0], 0.0, 1.5)
            x = np.linspace(0.0, 1.5, 0.05)
            y = x**2
            points = np.column_stack((x,y)).tolist()
            
            pass
        elif trajectory_type == 1:
            # TODO Part 6: Implement sigmoid
            # goal_x = np.clip(self.goal_point[0], 0.0, 2.5)
            x = np.linspace(0.0, 2.5, 0.05)
            y = 2 / (1+np.exp(-2*x)) - 1
            points = np.column_stack((x,y)).tolist()
            
        # the return should be a list of trajectory points: [ [x1,y1], ..., [xn,yn]]
        return points

