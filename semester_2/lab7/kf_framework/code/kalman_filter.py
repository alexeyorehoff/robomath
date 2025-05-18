import numpy as np
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data
from matplotlib.patches import Ellipse

# plot preferences, interactive plotting mode
fig = plt.figure()
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()


def plot_state(mu, sigma, landmarks, map_limits):
    # Visualizes the state of the kalman filter.
    #
    # Displays the mean and standard deviation of the belief,
    # the state covariance sigma and the position of the 
    # landmarks.

    # landmark positions
    lx = []
    ly = []

    for i in range(len(landmarks)):
        lx.append(landmarks[i + 1][0])
        ly.append(landmarks[i + 1][1])

    # mean of belief as current estimate
    estimated_pose = mu

    # calculate and plot covariance ellipse
    covariance = sigma[0:2, 0:2]
    eigenvals, eigenvecs = np.linalg.eig(covariance)

    # get largest eigenvalue and eigenvector
    max_ind = np.argmax(eigenvals)
    max_eigvec = eigenvecs[:, max_ind]
    max_eigval = eigenvals[max_ind]

    # get smallest eigenvalue and eigenvector
    min_ind = 0
    if max_ind == 0:
        min_ind = 1

    min_eigval = eigenvals[min_ind]

    # chi-square value for sigma confidence interval
    chi_square_scale = 2.2789

    # calculate width and height of confidence ellipse
    width = 2 * np.sqrt(chi_square_scale * max_eigval)
    height = 2 * np.sqrt(chi_square_scale * min_eigval)
    angle = np.arctan2(max_eigvec[1], max_eigvec[0])

    # generate covariance ellipse
    ell = Ellipse(xy=[estimated_pose[0], estimated_pose[1]], width=width, height=height, angle=angle / np.pi * 180)
    ell.set_alpha(0.25)

    # plot filter state and covariance
    plt.clf()
    plt.gca().add_artist(ell)
    plt.plot(lx, ly, 'bo', markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy', scale_units='xy')
    plt.axis(map_limits)
    
    plt.pause(0.01)


def prediction_step(odometry, mu, sigma):
    # Updates the belief, i.e., mu and sigma, according to the motion 
    # model
    # 
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution 
    
    x = mu[0]
    y = mu[1]
    theta = mu[2]

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    '''your code here'''

    # State prediction (motion model)
    new_x = x + delta_trans * np.cos(theta + delta_rot1)
    new_y = y + delta_trans * np.sin(theta + delta_rot1)
    new_theta = theta + delta_rot1 + delta_rot2
    
    mu = np.array([new_x, new_y, new_theta])

    # Jacobian of motion model (Gt)
    Gt = np.array([
        [1, 0, -delta_trans * np.sin(theta + delta_rot1)],
        [0, 1, delta_trans * np.cos(theta + delta_rot1)],
        [0, 0, 1]
    ])

    # Process noise covariance matrix
    Qt = np.array([
        [0.2, 0, 0],
        [0, 0.2, 0],
        [0, 0, 0.2]
    ])

    # Covariance prediction
    sigma = Gt @ sigma @ Gt.T + Qt

    '''***        ***'''

    return mu, sigma


def correction_step(sensor_data, mu, sigma, landmarks):
    # updates the belief, i.e., mu and sigma, according to the
    # sensor model
    # 
    # The employed sensor model is range-only
    #
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution 

    x = mu[0]
    y = mu[1]

    # measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']

    '''your code here'''
    theta = mu[2]
    bearings = sensor_data['bearing']

    Rt = np.eye(2 * len(ids)) * 0.5
    
    # Initialize matrices for stacked measurements, predictions, and Jacobians
    z = np.zeros(2 * len(ids))
    z_pred = np.zeros(2 * len(ids))
    H = np.zeros((2 * len(ids), 3))
    
    for i in range(len(ids)):
        lm_id = ids[i]
        lm_x = landmarks[lm_id][0]
        lm_y = landmarks[lm_id][1]
        
        # Actual measurement
        z[2*i] = ranges[i]
        z[2*i+1] = bearings[i]
        
        # Predicted measurement
        dx = lm_x - x
        dy = lm_y - y
        q = dx**2 + dy**2
        
        z_pred[2*i] = np.sqrt(q)
        z_pred[2*i+1] = np.arctan2(dy, dx) - theta
        
        # Jacobian for this measurement
        H[2*i:2*i+2, :] = np.array([
            [-dx/np.sqrt(q), -dy/np.sqrt(q), 0],
            [dy/q, -dx/q, -1]
        ])
    
    # Kalman gain
    K = sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + Rt)
    
    # State correction
    mu = mu + K @ (z - z_pred)
    mu[2] = np.arctan2(np.sin(mu[2]), np.cos(mu[2]))  # Normalize angle
    
    # Covariance correction
    sigma = (np.eye(3) - K @ H) @ sigma

    '''***        ***'''

    return mu, sigma


def main():
    # implementation of an extended Kalman filter for robot pose estimation

    print("Reading landmark positions")
    landmarks = read_world("../data/world.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data("../data/sensor_data.dat")

    # initialize belief
    mu = [0.0, 0.0, 0.0]
    sigma = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])

    map_limits = [-1, 12, -1, 10]

    # run kalman filter
    for timestep in range(len(sensor_readings) // 2):

        # plot the current state
        plot_state(mu, sigma, landmarks, map_limits)

        # perform prediction step
        mu, sigma = prediction_step(sensor_readings[timestep, 'odometry'], mu, sigma)

        # perform correction step
        mu, sigma = correction_step(sensor_readings[timestep, 'sensor'], mu, sigma, landmarks)

    plt.show(block=True)


if __name__ == "__main__":
    main()
