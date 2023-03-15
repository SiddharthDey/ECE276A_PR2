import numpy as np
import matplotlib.pyplot as plt
import os
import confuse
from pr2_utils import bresenham2D, mapCorrelation
from tqdm import tqdm

yaml_file = "configPR2.yaml"
config = confuse.load_yaml(yaml_file)

dataset = config["DATASET"]
  
with np.load(config["DATA_FOLDER"] + "Encoders%d.npz"%dataset) as data:
    encoder_counts = data["counts"] # 4 x n encoder counts
    encoder_stamps = data["time_stamps"] # encoder time stamps

with np.load(config["DATA_FOLDER"] + "Hokuyo%d.npz"%dataset) as data:
    lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
    lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
    lidar_range_min = data["range_min"] # minimum range value [m]
    lidar_range_max = data["range_max"] # maximum range value [m]
    lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans

with np.load(config["DATA_FOLDER"] + "Imu%d.npz"%dataset) as data:
    imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
    imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements

with np.load(config["DATA_FOLDER"] + "Kinect%d.npz"%dataset) as data:
    disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
    rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images


imu_closest_index = []
for encoder_stamp in encoder_stamps:
    imu_closest_index.append(np.argmin(abs(imu_stamps - encoder_stamp)))

imu_angular_velocity_closest = imu_angular_velocity[:,imu_closest_index]


x_particles_all = np.zeros((config["N_PARTICLES"], len(encoder_stamps)))
y_particles_all = np.zeros((config["N_PARTICLES"], len(encoder_stamps)))
theta_particles_all = np.zeros((config["N_PARTICLES"], len(encoder_stamps)))

for index in tqdm(range(1, len(encoder_stamps))):
    delta_t = (encoder_stamps[index] - encoder_stamps[index - 1])
    angular_velocity = imu_angular_velocity_closest[2, index]
    encoder_current = encoder_counts[:, index]
    linear_velocity = ((np.sum(encoder_current)/4.0)/(delta_t))*config["DISTANCE_PER_TIC"]

    linear_velocity_noise = np.random.normal(0, config["SIGMA_V"], config["N_PARTICLES"])
    angular_velocity_noise = np.random.normal(0, config["SIGMA_W"], config["N_PARTICLES"])

    linear_velocity_with_noise = linear_velocity + linear_velocity_noise
    angular_velocity_with_noise = angular_velocity + angular_velocity_noise

    x_prev_all = x_particles_all[:, index - 1]
    y_prev_all = y_particles_all[:, index - 1]
    theta_prev_all = theta_particles_all[:, index - 1]



    x_new_all = x_prev_all + delta_t*linear_velocity_with_noise*np.cos(theta_prev_all)
    y_new_all = y_prev_all + delta_t*linear_velocity_with_noise*np.sin(theta_prev_all)
    theta_new_all = theta_prev_all + delta_t*angular_velocity_with_noise

    x_particles_all[:, index] = x_new_all
    y_particles_all[:, index] = y_new_all
    theta_particles_all[:, index] = theta_new_all


plt.figure()
for index in range(config["N_PARTICLES"]):
    plt.plot(x_particles_all[index, :], y_particles_all[index, :])

plt.xlabel("x")
plt.ylabel("y")
save_path = config["PLOTS_SAVE_FOLDER"] + str(config["DATASET"]) \
                    + "_n_particles_trajectory.png"
plt.savefig(save_path)
plt.show()
plt.pause(10)