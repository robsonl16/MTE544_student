from rclpy.time import Time
from utilities import Logger

# Controller type
P=0 # poportional
PD=1 # proportional and derivative
PI=2 # proportional and integral
PID=3 # proportional, integral, derivative

class PID_ctrl:
    """
    One P/PD/PI/PID controller for a single error signal (one instance for linear, one for angular).
    Keeps a short history of stamped errors to estimate the derivative and integral, and logs
    e, e_dot, e_int and the time stamp to a CSV every update.
    """

    def __init__(self, type_, kp=1.2,kv=0.8,ki=0.2, history_length=3, filename_="errors.csv"):

        # Data for the controller: sliding window of the last `history_length` [error, stamp] pairs
        self.history_length=history_length
        self.history=[]
        self.type=type_

        # Controller gains
        self.kp=kp    # proportional gain
        self.kv=kv    # derivative gain
        self.ki=ki    # integral gain
        
        self.logger=Logger(filename_)
        # Remeber that you are writing to the file named filename_ or errors.csv the following:
            # error, error_dot, error_int and time stamp

    
    def update(self, stamped_error, status):
        """
        stamped_error = [error, ros_stamp]. Returns the control output (velocity command).
        status=False still updates the history/log but outputs 0 (robot held still).
        """

        if status == False:
            self.__update(stamped_error)
            return 0.0
        else:
            return self.__update(stamped_error)

        
    def __update(self, stamped_error):
        # Add the new error to the window, estimate e_dot and e_int, log, then apply the control law.

        latest_error=stamped_error[0]
        stamp=stamped_error[1]

        # Keep only the most recent history_length samples (sliding window)
        self.history.append(stamped_error)

        if (len(self.history) > self.history_length):
            self.history.pop(0)
        
        # If insufficient data points, use only the proportional gain
        if (len(self.history) != self.history_length):
            return self.kp * latest_error
        
        # Compute the error derivative
        dt_avg=0
        error_dot=0
        
        for i in range(1, len(self.history)):
            
            t0=Time.from_msg(self.history[i-1][1])
            t1=Time.from_msg(self.history[i][1])
            
            dt=(t1.nanoseconds - t0.nanoseconds) / 1e9
            
            dt_avg+=dt

            # use constant dt if the messages arrived inconsistent
            # for example dt=0.1 overwriting the calculation          
            
            # Part 5: finite-difference derivative between consecutive samples: (e_i - e_{i-1}) / dt
            error_dot += (self.history[i][0] - self.history[i-1][0]) / dt

        # Average the finite differences and the sample period over the window
        error_dot/=len(self.history)
        dt_avg/=len(self.history)

        # Compute the error integral: rectangle rule (sum of errors * dt) over the window only,
        # i.e. a windowed integral of the last history_length samples, not a running total
        sum_=0
        for hist in self.history:
            # Part 5: Gather the integration
            sum_+=hist[0]

        error_int=sum_*dt_avg
        
        # Part 4: Log your errors (e, e_dot, e_int, stamp in ns) for plotting/tuning
        self.logger.log_values([latest_error,
                                error_dot,
                                error_int,
                                Time.from_msg(stamp).nanoseconds])
        
        # Part 4: Implement the control law of P-controller: u = Kp*e
        if self.type == P:
            return self.kp * latest_error

        # Part 5: Implement the control law corresponding to each type of controller
        elif self.type == PD:
            # PD control law: u = Kp*e + Kd*e_dot   (C(s) = Kp + Kd*s)
            return self.kp * latest_error + self.kv * error_dot

        elif self.type == PI:
            # PI control law: u = Kp*e + Ki*e_int   (C(s) = Kp + Ki/s)
            return self.kp * latest_error + self.ki * error_int

        elif self.type == PID:
            # PID control law: u = Kp*e + Kd*e_dot + Ki*e_int   (C(s) = Kp + Kd*s + Ki/s)
            return self.kp * latest_error + self.kv * error_dot + self.ki * error_int