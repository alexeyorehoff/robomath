from read_data import read_world, read_sensor_data
from misc_tools import *
import numpy as np
import math
import copy

# plot preferences, interactive plotting mode
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()


def initialize_particles(num_particles, num_landmarks):
    # initialize particle at pose [0, 0, 0] with an empty map

    particles = []

    for _ in range(num_particles):
        particle = dict()

        # initialize pose: at the beginning, robot is certain it is at [0, 0, 0]
        particle['x'] = 0
        particle['y'] = 0
        particle['theta'] = 0

        # initial weight
        particle['weight'] = 1.0 / num_particles

        # particle history aka all visited poses
        particle['history'] = []

        # initialize landmarks of the particle
        landmarks = dict()

        for i in range(num_landmarks):
            landmark = dict()

            # initialize the landmark mean and covariance
            landmark['mu'] = [0, 0]
            landmark['sigma'] = np.zeros([2, 2])
            landmark['observed'] = False

            landmarks[i + 1] = landmark

        # add landmarks to particle
        particle['landmarks'] = landmarks

        # add particle to set
        particles.append(particle)

    return particles


def sample_motion_model(odometry, particles):
    # Updates the particle positions, based on old positions, the odometry
    # measurements and the motion noise 

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.05, 0.05]

    '''your code here'''
    for particle in particles:
        # Generate noisy motion parameters
        sigma_rot1 = math.sqrt(noise[0] * delta_rot1**2 + noise[1] * delta_trans**2)
        sigma_trans = math.sqrt(noise[2] * delta_trans**2 + noise[3] * (delta_rot1**2 + delta_rot2**2))
        sigma_rot2 = math.sqrt(noise[0] * delta_rot2**2 + noise[1] * delta_trans**2)
        
        noisy_rot1 = delta_rot1 + np.random.normal(0, sigma_rot1)
        noisy_trans = delta_trans + np.random.normal(0, sigma_trans)
        noisy_rot2 = delta_rot2 + np.random.normal(0, sigma_rot2)
        
        # Update particle pose
        x = particle['x']
        y = particle['y']
        theta = particle['theta']
        
        new_x = x + noisy_trans * math.cos(theta + noisy_rot1)
        new_y = y + noisy_trans * math.sin(theta + noisy_rot1)
        new_theta = (theta + noisy_rot1 + noisy_rot2 + math.pi) % (2 * math.pi) - math.pi
        
        particle['x'] = new_x
        particle['y'] = new_y
        particle['theta'] = new_theta
        particle['history'].append([new_x, new_y, new_theta])
    '''***        ***'''


def measurement_model(particle, landmark):
    # Compute the expected measurement for a landmark
    # and the Jacobian with respect to the landmark.

    px = particle['x']
    py = particle['y']
    p_theta = particle['theta']

    lx = landmark['mu'][0]
    ly = landmark['mu'][1]

    # calculate expected range measurement
    meas_range_exp = np.sqrt((lx - px) ** 2 + (ly - py) ** 2)
    meas_bearing_exp = math.atan2(ly - py, lx - px) - p_theta

    h = np.array([meas_range_exp, meas_bearing_exp])

    # Compute the Jacobian h_j of the measurement function h
    # wrt the landmark location

    h_j = np.zeros((2, 2))
    h_j[0, 0] = (lx - px) / h[0]
    h_j[0, 1] = (ly - py) / h[0]
    h_j[1, 0] = (py - ly) / (h[0] ** 2)
    h_j[1, 1] = (lx - px) / (h[0] ** 2)

    return h, h_j


def eval_sensor_model(sensor_data, particles):
    # Correct landmark poses with a measurement and
    # calculate particle weight

    # sensor noise
    q_t = np.array([[0.1, 0],
                    [0, 0.1]])

    # measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    bearings = sensor_data['bearing']

    # update landmarks and calculate weight for each particle
    for particle in particles:

        landmarks = particle['landmarks']
        particle['weight'] = 1.0

        px = particle['x']
        py = particle['y']
        p_theta = particle['theta']

        # loop over observed landmarks
        for i in range(len(ids)):

            # current landmark
            lm_id = ids[i]
            landmark = landmarks[lm_id]

            # measured range and bearing to current landmark
            meas_range = ranges[i]
            meas_bearing = bearings[i]

            if not landmark['observed']:
                # landmark is observed for the first time
                '''your code here'''
                # Initialize landmark position
                landmark['mu'] = [
                    px + meas_range * math.cos(p_theta + meas_bearing),
                    py + meas_range * math.sin(p_theta + meas_bearing)
                ]
                
                # Compute Jacobian
                _, H = measurement_model(particle, landmark)
                
                # Initialize covariance
                landmark['sigma'] = np.linalg.inv(H) @ q_t @ np.linalg.inv(H).T
                landmark['observed'] = True
                '''***        ***'''

            else:
                # landmark was observed before
                '''your code here'''
                # Predict measurement
                z_pred, H = measurement_model(particle, landmark)
                
                # Compute innovation
                z_actual = np.array([meas_range, meas_bearing])
                innovation = z_actual - z_pred
                innovation[1] = (innovation[1] + math.pi) % (2 * math.pi) - math.pi  # Normalize angle
                
                # Compute innovation covariance
                S = H @ landmark['sigma'] @ H.T + q_t
                
                # Kalman gain
                K = landmark['sigma'] @ H.T @ np.linalg.inv(S)
                
                # Update landmark mean and covariance
                landmark['mu'] = (np.array(landmark['mu']) + K @ innovation).tolist()
                landmark['sigma'] = (np.eye(2) - K @ H) @ landmark['sigma']
                
                # Update particle weight
                particle['weight'] *= math.exp(-0.5 * innovation.T @ np.linalg.inv(q_t) @ innovation) / \
                                     math.sqrt(np.linalg.det(2 * math.pi * q_t))
                '''***        ***'''

    # normalize weights
    normalizer = sum([p['weight'] for p in particles])

    for particle in particles:
        particle['weight'] = particle['weight'] / normalizer


def resample_particles(particles):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle 
    # weights.

    new_particles = []

    '''your code here'''
    num_particles = len(particles)
    weights = np.array([p['weight'] for p in particles])
    
    # Generate random starting point
    step = 1.0 / num_particles
    u = np.random.uniform(0, step)
    
    # Compute cumulative weights
    cum_weights = np.cumsum(weights)
    
    i = 0
    for j in range(num_particles):
        # Move along the weight distribution
        while u > cum_weights[i]:
            i += 1
        
        # Add a copy of the selected particle
        new_particle = copy.deepcopy(particles[i])
        new_particle['weight'] = 1.0 / num_particles  # Reset weight
        new_particles.append(new_particle)
        
        # Move to next position
        u += step
    '''***        ***'''

    return new_particles


def main():

    print("Reading landmark positions")
    landmarks = read_world("../data/world.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data("../data/sensor_data.dat")

    num_particles = 100
    num_landmarks = len(landmarks)

    # create particle set
    particles = initialize_particles(num_particles, num_landmarks)

    # run FastSLAM
    for timestep in range(len(sensor_readings) // 2):

        # predict particles by sampling from motion model with odometry info
        sample_motion_model(sensor_readings[timestep, 'odometry'], particles)

        # evaluate sensor model to update landmarks and calculate particle weights
        eval_sensor_model(sensor_readings[timestep, 'sensor'], particles)

        # plot filter state
        plot_state(particles, landmarks)

        # calculate new set of equally weighted particles
        particles = resample_particles(particles)

    plt.show(block=True)


if __name__ == "__main__":
    main()