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


def clean_lidar_scan(input_ranges, input_angles, range_min, range_max):
    indValid = np.logical_and((input_ranges < range_max),(input_ranges > range_min))
    ranges = input_ranges[indValid]
    angles = input_angles[indValid]

    return ranges, angles

def clean_lidar_scan_only(input_ranges, range_min, range_max):
    indValid = np.logical_and((input_ranges < range_max),(input_ranges > range_min))
    ranges = input_ranges[indValid]

    return ranges

def cartesian_to_map_index(cartesian_x, cartesian_y, MAP):
    index_x = ((cartesian_x - MAP['xmin'])/MAP['res']).astype(int)
    index_y = ((cartesian_y - MAP['ymin'])/MAP['res']).astype(int)

    return index_x, index_y


MAP = {}
MAP['res']   = config["MAP_RES"] #meters
MAP['xmin']  =  config["MAP_X_MIN"]  #meters
MAP['ymin']  =  config["MAP_Y_MIN"]
MAP['xmax']  =  config["MAP_X_MAX"]
MAP['ymax']  =  config["MAP_Y_MAX"]
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey'])) #DATA TYPE: char or int8

lidar_angles_all = np.linspace(lidar_angle_min, lidar_angle_max, np.shape(lidar_ranges)[0])

for index in tqdm(range(1, len(encoder_stamps))):
    lidar_scan = lidar_ranges_closest[:, index]
    lidar_scan, lidar_angles = clean_lidar_scan(lidar_scan, lidar_angles_all, lidar_range_min, lidar_range_max)

    local_sensor_x = lidar_scan * np.cos(lidar_angles)
    local_sensor_y = lidar_scan * np.sin(lidar_angles)

    local_body_x = local_sensor_x + config["X_OFFSET"]
    local_body_y = local_sensor_y

    x_current = x_all[index]
    y_current = y_all[index]
    theta_current = theta_all[index]

    p = np.array([[x_current, y_current]]).T
    R = np.array([[np.cos(theta_current), -np.sin(theta_current)],
                    [np.sin(theta_current), np.cos(theta_current)]])

    local_body_xy = np.array([[local_body_x, local_body_y]]).T
    global_xy = R@local_body_xy + p

    global_x = global_xy[:, 0, 0]
    global_y = global_xy[:, 1, 0]

    global_x_ind = (global_x > -config["MAP_GRID_METERS"]) & (global_x < config["MAP_GRID_METERS"])
    global_y_ind = (global_y > -config["MAP_GRID_METERS"]) & (global_y < config["MAP_GRID_METERS"])
    global_xy_ind = global_x_ind & global_y_ind

    global_x = global_x[global_xy_ind]
    global_y = global_y[global_xy_ind]

    if x_current > config["MAP_GRID_METERS"] or x_current < -config["MAP_GRID_METERS"] or \
                y_current > config["MAP_GRID_METERS"] or y_current < -config["MAP_GRID_METERS"]:
        continue

    lidar_x_index, lidar_y_index = cartesian_to_map_index(global_x, global_y, MAP)
    global_bot_x, global_bot_y = cartesian_to_map_index(x_current, y_current, MAP)


    for lidar_index in range(len(lidar_x_index)):

        lidar_x_index_current = lidar_x_index[lidar_index]
        lidar_y_index_current = lidar_y_index[lidar_index]

        MAP['map'][lidar_x_index_current, lidar_y_index_current] += 2*np.log(4)

        free_cells = bresenham2D(global_bot_x, global_bot_y, lidar_x_index_current, lidar_y_index_current)
        free_cells = free_cells.astype(int)
        MAP['map'][np.array(free_cells[0]), np.array(free_cells[1])] -= np.log(4)


def get_binary_map(map, THRESHOLD):
    binary_map = np.exp(MAP['map']) / (1 + np.exp(MAP['map']))

    binary_map = binary_map > THRESHOLD

    return binary_map.astype(int)

binary_map = get_binary_map(MAP['map'], 0.1)


plt.imshow(binary_map)
save_path = config["PLOTS_SAVE_FOLDER"] + str(config["DATASET"]) \
                    + "_map_dead_reckoning.png"
plt.savefig(save_path)
plt.show()
plt.pause(10)
    