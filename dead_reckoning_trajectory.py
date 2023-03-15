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

lidar_closest_index = []
for encoder_stamp in encoder_stamps:
    lidar_closest_index.append(np.argmin(abs(lidar_stamps - encoder_stamp)))

lidar_ranges_closest = lidar_ranges[:,lidar_closest_index]

x_all = [0]
y_all = [0]
theta_all = [0]

for index in tqdm(range(1, len(encoder_stamps))):
    delta_t = (encoder_stamps[index] - encoder_stamps[index - 1])
    yaw_velocity = imu_angular_velocity_closest[2, index]
    encoder_current = encoder_counts[:, index]
    linear_velocity = ((np.sum(encoder_current)/4.0)/(delta_t))*config["DISTANCE_PER_TIC"]

    x_prev = x_all[-1]
    y_prev = y_all[-1]
    theta_prev = theta_all[-1]

    x_new = x_prev + delta_t*linear_velocity*np.cos(theta_prev)
    y_new = y_prev + delta_t*linear_velocity*np.sin(theta_prev)
    theta_new = theta_prev + delta_t*yaw_velocity

    x_all.append(x_new)
    y_all.append(y_new)
    theta_all.append(theta_new)

# plt.figure()
plt.plot(theta_all)
plt.xlabel("time stamp")
plt.ylabel("theta")
save_path = config["PLOTS_SAVE_FOLDER"] + str(config["DATASET"]) \
                    + "_theta_dead_reckoning.png"
plt.savefig(save_path)
plt.show()
plt.pause(10)


# plt.figure()
plt.plot(x_all, y_all)
plt.xlabel("x")
plt.ylabel("y")
save_path = config["PLOTS_SAVE_FOLDER"] + str(config["DATASET"]) \
                    + "_trajectory_dead_reckoning.png"
plt.savefig(save_path)
plt.show()
plt.pause(10)

    
    