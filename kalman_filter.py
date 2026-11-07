import numpy as np


# TODO Part 3: Comment the code explaining each part
class kalman_filter:
    # Part 3: Initialize the covariances and the states
    # Do this in lab by looking at covariance of the sensors and the states
    def __init__(self, P, Q, R, x, dt):

        self.P = P  # State covariance matrix
        self.Q = Q  # Process noise covariance matrix
        self.R = R  # Measurement model covariance matrix
        self.x = x  # States
        self.dt = dt

    # Part 3: Replace the matrices with Jacobians where needed
    def predict(self):
        """
        need to run the motion model to get new state,
        then after that calculate the jacobian of the motion model
        to get A, then update P
        """
        self.A = self.jacobian_A()  # calculate jacobian for motion model
        self.motion_model()  # run the motion model for one step
        self.C = self.jacobian_H()  # calculate jacobian for measurement model

        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    # Part 3: Calculate the Kalman gain and correct your estimates
    def update(self, z):
        """
        update step involving calculating the Kalman gain
        then updating the state
        then updating the covariance matrix
        """
        # predicted measurement and how far it was
        innovation = z - self.measurement_model()

        # innovation covariance: predicted uncertainty in measurement + sensor noise
        S = self.C @ self.P @ self.C.T + self.R

        kalman_gain = self.P @ self.C.T @ np.linalg.inv(S)

        self.x = self.x + kalman_gain @ innovation

        # reduce uncertainty to account for measurement
        self.P = np.array(np.eye(len(self.x)) - kalman_gain @ self.C) @ self.P

    # Part 3: Implement here the measurement model
    def measurement_model(self):
        x, y, th, w, v, vdot = self.x
        return np.array(
            [
                v,  # v
                w,  # w
                vdot,  # ax
                v * w,  # ay
            ]
        )

    # Part 3: Impelment the motion model (state-transition matrice)
    def motion_model(self):
        """
        How the 2wd robot moves between timesteps, given a current state and time step
        """
        x, y, th, w, v, vdot = self.x
        dt = self.dt

        self.x = np.array(
            [
                x + v * np.cos(th) * dt,
                y + v * np.sin(th) * dt,
                th + w * dt,
                w,
                v + vdot * dt,
                vdot,
            ]
        )

    def jacobian_A(self):
        """
        Calculate the jacobian matrix of the motion model wrt to the state variables
        The motion model is:
        x_{k+1} = x_k + v_k * cos(th_k) * dt
        y_{k+1} = y_k + v_k * sin(th_k) * dt
        th_{k+1} = th_k + w_k * dt
        w_{k+1} = w_k
        v_{k+1} = v_k + vdot_k * dt
        vdot_{k+1} = vdot_k
        """
        x, y, th, w, v, vdot = self.x
        dt = self.dt

        return np.array(
            [
                # x, y,  th,                      w,    v,             vdot
                [1, 0, -v * np.sin(th) * dt, 0, np.cos(th) * dt, 0],
                [0, 1, v * np.cos(th) * dt, 0, np.sin(th) * dt, 0],
                [0, 0, 1, dt, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, dt],
                [0, 0, 0, 0, 0, 1],
            ]
        )

    # Part 3: Implement here the jacobian of the H matrix (measurements)
    def jacobian_H(self):
        x, y, th, w, v, vdot = self.x
        """
        Calculate the jacobian matrix of the measurement model
        measurment model: z = [v, w, ax, ay] where ax is linear acceleration in x 
        and ay is linear acceleration in y, calculated with v*w
        In the Jacobian below, each row is the partial derivative of the measurement model
        wrt to each state variable (x, y, th, w, v, vdot)
        """
        return np.array(
            [
                # x, y, th,w, v, vdot
                [0, 0, 0, 0, 1, 0],  # v
                [0, 0, 0, 1, 0, 0],  # w
                [0, 0, 0, 0, 0, 1],  # ax
                [0, 0, 0, v, w, 0],  # ay
            ]
        )

    # Part 3: return the states here
    def get_states(self):
        return self.x
