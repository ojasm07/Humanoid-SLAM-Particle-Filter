import os, sys, pickle, math
from copy import deepcopy

from scipy import io
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    """
    This will maintain the occupancy grid and log_odds. You do not need to change anything
    in the initialization
    """
    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.xmin, s.xmax = -20, 20
        s.ymin, s.ymax = -20, 20
        s.szx = int(np.ceil((s.xmax-s.xmin)/s.resolution+1))
        s.szy = int(np.ceil((s.ymax-s.ymin)/s.resolution+1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szy), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received yet for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh/(1-s.occupied_prob_thresh))
        
    def grid_cell_from_xy(s, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/y go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        #### TODO: XXXXXXXXXXX
        # raise NotImplementedError
        
        cells = np.vstack((
        np.ceil((np.clip(x, s.xmin, s.xmax) - s.xmin) / s.resolution).astype(np.int16),
        np.ceil((np.clip(y, s.ymin, s.ymax) - s.ymin) / s.resolution).astype(np.int16)
                ))
        
        return cells

class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.05, Q=1e-3*np.eye(3),
                 resampling_threshold=0.3):
        s.init_sensor_model()

        # dynamics noise for the state (x,y,yaw)
        s.Q = 1e-8*np.eye(3)

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

    def read_data(s, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar = load_lidar_data(os.path.join(src_dir,
                                               'data/%s/%s_lidar%d'%(split,split,idx)))
        s.joint = load_joint_data(os.path.join(src_dir,
                                               'data/%s/%s_joint%d'%(split,split,idx)))

        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t
        s.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(s.joint['t']-t))

    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.head_height = 0.93 + 0.33
        s.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        s.lidar_dmin = 1e-3
        s.lidar_dmax = 30
        s.lidar_angular_resolution = 0.25
        # these are the angles of the rays of the Hokuyo
        s.lidar_angles = np.arange(-135,135+s.lidar_angular_resolution,
                                   s.lidar_angular_resolution)*np.pi/180.0

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3,s.n), dtype=np.float64)
        s.w = deepcopy(w) if w is not None else np.ones(n)/float(s.n)

    @staticmethod
    def stratified_resampling(p, w):
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """
        #### TODO: XXXXXXXXXXX
        # raise NotImplementedError

        # change variable names
        n = p.shape[1]
        noise = np.random.uniform(0,1/n)

        cumulative_weights = np.cumsum(w)
        indices = np.searchsorted(cumulative_weights, noise+np.arange(n)/n, side='right')
        new_wts = w[indices]
        new_p = p[:,indices]

        return new_p, new_wts

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def rays2world(s, p, d, head_angle=0, neck_angle=0, angles=None, t=0):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually
        be simply s.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, for each ray (the length of d has to be equal to that of angles, this is s.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        """
        #change variables and logic if possible

        #### TODO: XXXXXXXXXXX
        # raise NotImplementedError

        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading
        # the data
        d = np.clip(d, s.lidar_dmin, s.lidar_dmax)

        # 1. from lidar distances to points in the LiDAR frame
        lidar_distance = np.array([d*np.cos(angles), d*np.sin(angles), np.zeros(len(d)), np.ones(len(d))])

        # 2. from LiDAR frame to the body frame
        translation_b2h = np.array([0, 0, 0.33])
        # print("translation_b2h", translation_b2h.shape)
        lidar_to_head_rot_so3 = euler_to_so3(0, 0, 0)
        lidar_to_head_rot = np.vstack((np.hstack((lidar_to_head_rot_so3, translation_b2h.reshape(-1,1))), np.array([0,0,0,1])))
        head_to_body_rot = euler_to_se3(0, head_angle, neck_angle, translation_b2h)

        # 3. from body frame to world frame
        translation_2 = np.array([p[0], p[1], 0.93])
        # print("translation_2", translation_2.shape)
        # print(head_to_body_rot.shape, lidar_to_head_rot.shape)
        lidar_to_world_frame_so3 = euler_to_so3(*s.lidar[t]['rpy'])
        lidar_to_world_frame_rot = np.vstack((np.hstack((lidar_to_world_frame_so3, translation_2.reshape(-1,1))), np.array([0,0,0,1]))) @ head_to_body_rot @ lidar_to_head_rot
        world_coords = np.dot(lidar_to_world_frame_rot, lidar_distance)[:2, d > s.lidar_dmin]

        return world_coords

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two poses and we will simply set this to be the control (delta x, delta y, delta theta)
        """

        if t == 0:
            return np.zeros(3)

        #### TODO: XXXXXXXXXXX
        # raise NotImplementedError
        lidar_data_current = s.lidar[t]
        lidar_data_previous = s.lidar[t-1]
        # print('lidar_data_current', lidar_data_current['xyth'].shape)
        # print('lidar_data_previous', lidar_data_previous['xyth'].shape)
        def get_pose(lidar_data):
            return np.array([lidar_data['xyth'][0], lidar_data['xyth'][1], lidar_data['rpy'][2]])

        current_pose = get_pose(lidar_data_current)
        # print('current_pose', current_pose.shape)
        previous_pose = get_pose(lidar_data_previous)
        control_t = smart_minus_2d(current_pose, previous_pose)

        return control_t

    def dynamics_step(s, t):
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """
        #### TODO: XXXXXXXXXXX
        # raise NotImplementedError
        #smart_plus_2d takes 3x1 vecs
        n = s.p.shape[1]
        delta_pose = s.get_control(t)
        mean_t = np.zeros(3)
        noise_samples = np.random.multivariate_normal(mean=mean_t, cov=s.Q, size=n).T
        # s.p = smart_plus_2d(s.p, delta_pose[:, np.newaxis] + noise)
        for i in range(n):
            s.p[:,i] = smart_plus_2d(s.p[:,i], delta_pose+noise_samples[:,i])

    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        #### TODO: XXXXXXXXXXX
        # raise NotImplementedError
        prev_wts = np.log(w) + obs_logp
        wmax = prev_wts.max()
        log_softmax = np.log(np.exp(prev_wts-wmax).sum())
        new_wts = np.exp(prev_wts - wmax - log_softmax)
        return new_wts

    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data

        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle)
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight, and use its occupied cells to update the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration (it is simply the binarized version of log_odds). map.log_odds is of course maintained across iterations.
        """
        #### TODO: XXXXXXXXXXX
        # raise NotImplementedError

        angle = s.joint['head_angles'][:][s.find_joint_t_idx_from_lidar(t)]
        head_angle_t = angle[1]
        neck_angle_t = angle[0]
        lidar_rpy_t = s.lidar[t]['rpy']
        imu_data = s.lidar[t]['rpy']
        pose = s.map.grid_cell_from_xy(*(s.lidar[t]['xyth'][:2])).reshape(2, 1)
        space = lambda start, end, num: np.vstack((np.linspace(start[0], end[0], num, endpoint=False, dtype=int).T, np.linspace(start[1], end[1], num, endpoint=False, dtype=int).T))

        if t == 0:
            particle_world_coordinates = s.rays2world(s.p.T[0], s.lidar[t]['scan'], head_angle_t, neck_angle_t, s.lidar_angles)
            particle_grid = s.map.grid_cell_from_xy(*s.p.T[0, :2])
            occupied_c = s.map.grid_cell_from_xy(particle_world_coordinates[0], particle_world_coordinates[1])
            s.map.cells[occupied_c[0], occupied_c[1]] = 1

            c11 = deepcopy(particle_grid)
            for i in occupied_c.T:
                c22 = space(particle_grid, i, int(np.linalg.norm(i - particle_grid.T)))
                cell_f = np.hstack((c11, c22))
            cell_f = np.unique(cell_f, axis=1) 
            s.map.cells.fill(0)
            s.map.cells[occupied_c[0], occupied_c[1]] = 1
            particle_grid = s.map.grid_cell_from_xy(s.p.T[0, 0], s.p.T[0, 1]).reshape(2, 1)
            s.map.cells[cell_f[0], cell_f[1]] = 0

        else:

            log_of_obs = np.zeros(s.p.shape[1])
            for i in range(s.p.shape[1]):
                particle_world_coordinates = s.rays2world(s.p.T[i], s.lidar[t]['scan'], head_angle_t, neck_angle_t, s.lidar_angles, t)
                occupied_world_coords = s.map.grid_cell_from_xy(particle_world_coordinates[0], particle_world_coordinates[1])
                log_of_obs[i] = np.sum(s.map.cells[occupied_world_coords[0], occupied_world_coords[1]])
            s.w = s.update_weights(s.w, log_of_obs)

            likely_part = s.p.T[np.argmax(s.w)]
            particle_world_coordinates = s.rays2world(likely_part, s.lidar[t]['scan'], head_angle_t, neck_angle_t, s.lidar_angles, t)
            likely_occ = s.map.grid_cell_from_xy(particle_world_coordinates[0], particle_world_coordinates[1])
            particle_grid = s.map.grid_cell_from_xy(likely_part[0], likely_part[1])
            not_occupied_cell = deepcopy(particle_grid)
            for i in occupied_world_coords.T:
                c22 = space(particle_grid, i, int(np.linalg.norm(i - particle_grid.T)))
                not_occupied_cell = np.hstack((not_occupied_cell, c22))
            not_occupied_cell = np.unique(not_occupied_cell, axis=1)

            s.map.log_odds[not_occupied_cell[0], not_occupied_cell[1]] += s.lidar_log_odds_free
            s.map.log_odds[likely_occ[0], likely_occ[1]] += s.lidar_log_odds_occ
            np.clip(s.map.log_odds, -s.map.log_odds_max, s.map.log_odds_max, out=s.map.log_odds)
            cell_f = np.less_equal(s.map.log_odds, s.lidar_log_odds_free)
            occupied_cells = np.greater_equal(s.map.log_odds, s.map.log_odds_thresh)
            s.map.cells = np.zeros_like(s.map.cells)      
            s.map.cells[occupied_cells] = 1
            s.map.cells[cell_f] = 0
            s.map.cells[np.logical_not(np.logical_or(cell_f, occupied_cells))] = 2
            # s.resample_particles(s)
            s.resample_particles()

        return pose, particle_grid


    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')