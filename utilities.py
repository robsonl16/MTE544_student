from math import atan2, asin, sqrt

M_PI=3.1415926535

class Logger:
    """Minimal CSV logger: writes the header on creation, then appends one row per log_values() call."""

    def __init__(self, filename, headers=["e", "e_dot", "e_int", "stamp"]):

        self.filename = filename

        # Create/overwrite the file and write the comma-separated header row

        with open(self.filename, 'w') as file:
            
            header_str=""

            for header in headers:
                header_str+=header
                header_str+=", "
            
            header_str+="\n"
            
            file.write(header_str)


    def log_values(self, values_list):
        # Append one row; the file is reopened each time so data is saved even if the node is killed

        with open(self.filename, 'a') as file:
            
            vals_str=""
            
            for value in values_list:
                vals_str+=f"{value}, "
            
            vals_str+="\n"
            
            file.write(vals_str)
            

    def save_log(self):
        # Nothing to do: log_values() already writes every row to disk
        pass

class FileReader:
    """Reads a Logger CSV back into (headers, rows of floats); used by plot_errors.py."""
    def __init__(self, filename):
        
        self.filename = filename
        
        
    def read_file(self):
        
        read_headers=False

        table=[]
        headers=[]
        with open(self.filename, 'r') as file:

            if not read_headers:
                for line in file:
                    values=line.strip().split(',')

                    for val in values:
                        if val=='':
                            break
                        headers.append(val.strip())

                    read_headers=True
                    break
            
            next(file)
            
            # Read each line and extract values
            for line in file:
                values = line.strip().split(',')
                
                row=[]                
                
                for val in values:
                    if val=='':
                        break
                    row.append(float(val.strip()))

                table.append(row)
        
        return headers, table
    
    

def euler_from_quaternion(quat):
    """
    Convert quaternion (w in last place) to yaw only (planar robot, roll/pitch not needed).
    quat = [x, y, z, w]; returns yaw in [-pi, pi].
    """

    # just unpack yaw
    x,y,z,w = quat
    yaw = atan2(2*(w*z + x*y), 1-2*(y**2 + z**2))
    return yaw


# Part 4: Implement the calculation of the linear error
def calculate_linear_error(current_pose, goal_pose):
    # Linear error = Euclidean distance from robot to goal (always >= 0).
    # current_pose = [x, y, theta, stamp], goal_pose = [x, y]
    error_linear= sqrt((goal_pose[0]-current_pose[0])**2 + (goal_pose[1]-current_pose[1])**2)

    return error_linear

# Part 4: Implement the calculation of the angular error
def calculate_angular_error(current_pose, goal_pose):
    # Angular error = (bearing from robot to goal) - (robot heading), wrapped to [-pi, pi].
    # Positive = goal is to the robot's left (turn CCW). current_pose = [x, y, theta, stamp]

    # Desired heading: direction of the vector robot -> goal
    desired_theta = atan2(
        goal_pose[1] - current_pose[1],
        goal_pose[0] - current_pose[0],
    )
    current_theta = current_pose[2]

    error_angular = desired_theta - current_theta

    # Wrap to [-pi, pi] so the robot always turns the short way
    while error_angular > M_PI:
        error_angular -= 2 * M_PI

    while error_angular < -M_PI:
        error_angular += 2 * M_PI

    return error_angular
