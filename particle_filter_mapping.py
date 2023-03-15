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


MAP = {}
MAP['res']   = config["MAP_RES"] #meters
MAP['xmin']  =  config["MAP_X_MIN"]  #meters
MAP['ymin']  =  config["MAP_Y_MIN"]
MAP['xmax']  =  config["MAP_X_MAX"]
MAP['ymax']  =  config["MAP_Y_MAX"]
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey'])) #DATA TYPE: char or int8

lidar_angles_all = np.linspace(lidar_angle_min, lidar_angle_max, len(lidar_ranges))

def get_binary_map(map, THRESHOLD):
    binary_map = np.exp(MAP['map']) / (1 + np.exp(MAP['map']))

    binary_map = binary_map > THRESHOLD

    return binary_map.astype(int)

x_best_all = [0]
y_best_all = [0]
theta_best_all = [0]

x_current_all = np.zeros((config["N_PARTICLES"]))
y_current_all = np.zeros((config["N_PARTICLES"]))
theta_current_all = np.zeros((config["N_PARTICLES"]))

x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map

x_range = np.arange(-MAP['res'], 1.5*MAP['res'], MAP['res'])
y_range = np.arange(-MAP['res'], 1.5*MAP['res'], MAP['res'])

alpha_current = np.ones((config["N_PARTICLES"]))
alpha_current = alpha_current/np.sum(alpha_current)

def clean_lidar_scan(input_ranges, input_angles, range_min, range_max):
    indValid = np.logical_and((input_ranges < range_max),(input_ranges > range_min))
    ranges = input_ranges[indValid]
    angles = input_angles[indValid]

    return ranges, angles

def cartesian_to_map_index(cartesian_x, cartesian_y, MAP):
    index_x = ((cartesian_x - MAP['xmin'])/MAP['res']).astype(int)
    index_y = ((cartesian_y - MAP['ymin'])/MAP['res']).astype(int)

    return index_x, index_y


for index in tqdm(range(1, len(encoder_stamps))):
    delta_t = (encoder_stamps[index] - encoder_stamps[index - 1])
    angular_velocity = imu_angular_velocity_closest[2, index]
    encoder_current = encoder_counts[:, index]
    linear_velocity = ((np.sum(encoder_current)/4.0)/(delta_t))*config["DISTANCE_PER_TIC"]

    lidar_scan = lidar_ranges_closest[:, index]
    lidar_scan, lidar_angles = clean_lidar_scan(lidar_scan, lidar_angles_all, lidar_range_min, lidar_range_max)
    local_sensor_x = lidar_scan * np.cos(lidar_angles)
    local_sensor_y = lidar_scan * np.sin(lidar_angles)
    local_body_x = local_sensor_x + config["X_OFFSET"]
    local_body_y = local_sensor_y

    linear_velocity_noise = np.random.normal(0, config["SIGMA_V"], config["N_PARTICLES"])
    angular_velocity_noise = np.random.normal(0, config["SIGMA_W"], config["N_PARTICLES"])

    linear_velocity_with_noise = linear_velocity + linear_velocity_noise
    angular_velocity_with_noise = angular_velocity + angular_velocity_noise

    x_new_all = x_current_all + delta_t*linear_velocity_with_noise*np.cos(theta_current_all)
    y_new_all = y_current_all + delta_t*linear_velocity_with_noise*np.sin(theta_current_all)
    theta_new_all = theta_current_all + delta_t*angular_velocity_with_noise

    lidar_local_all = np.array([local_body_x, local_body_y, np.ones((len(local_body_y)))])

    transformation_particle_all = np.array([[np.cos(theta_new_all), -np.sin(theta_new_all), x_new_all],
                                            [np.sin(theta_new_all), np.cos(theta_new_all), y_new_all],
                                            [np.zeros((len(theta_new_all))), np.zeros((len(theta_new_all))), np.ones((len(theta_new_all)))]])
    
    transformation_particle_all = np.transpose(transformation_particle_all, (2,0,1))

    lidar_global_all = transformation_particle_all@lidar_local_all

    c_all = np.zeros(([config["N_PARTICLES"]]))
    binary_map = get_binary_map(MAP['map'], 0.5)

    for particle_index in range(config["N_PARTICLES"]):
        lidar_global_particle = lidar_global_all[particle_index,:,:]
        c = mapCorrelation(binary_map, x_im, y_im, lidar_global_particle, x_range, y_range)

        c_argmax_x, c_argmax_y = np.unravel_index(c.argmax(), c.shape)

        c_all[particle_index] = np.max(c)

    best_particle_index = np.argmax(c_all)
    x_best = x_new_all[best_particle_index]
    y_best = y_new_all[best_particle_index]
    theta_best = theta_new_all[best_particle_index]

    lidar_global_best = lidar_global_all[best_particle_index, :, :]
    global_x = lidar_global_best[0,:]
    global_y = lidar_global_best[1,:]

    global_x_ind = (global_x > -config["MAP_GRID_METERS"]) & (global_x < config["MAP_GRID_METERS"])
    global_y_ind = (global_y > -config["MAP_GRID_METERS"]) & (global_y < config["MAP_GRID_METERS"])
    global_xy_ind = global_x_ind & global_y_ind

    lidar_global_x = global_x[global_xy_ind]
    lidar_global_y = global_y[global_xy_ind]

    lidar_x_map, lidar_y_map = cartesian_to_map_index(lidar_global_x, lidar_global_y, MAP)
    global_bot_x_map, global_bot_y_map = cartesian_to_map_index(x_best, y_best, MAP)
    
    for lidar_index in range(len(lidar_x_map)):
        lidar_x_map_current = lidar_x_map[lidar_index]
        lidar_y_map_current = lidar_y_map[lidar_index]
        MAP['map'][lidar_x_map_current, lidar_y_map_current] += 2*np.log(4)

        free_cells = bresenham2D(global_bot_x_map, global_bot_y_map, lidar_x_map_current, lidar_y_map_current)
        free_cells = free_cells.astype(int)
        MAP['map'][np.array(free_cells[0]), np.array(free_cells[1])] -= np.log(4)

    if index %100 == 0:
        binary_map = get_binary_map(MAP['map'], 0.1)
        plt.imshow(binary_map)
        save_path = config["PLOTS_SAVE_FOLDER"] + str(config["DATASET"]) \
                            + "_PF_map_updates.png"
        plt.savefig(save_path)

    if (c_all == 0.0).all():
        c_all = np.ones((len(c_all)))
        c_normalized = c_all/np.sum(c_all)
    else:
        c_normalized = c_all/np.sum(c_all)

    alpha_new = alpha_current*c_normalized
    alpha_new = alpha_new/np.sum(alpha_new)
    
    N_eff = 1/(np.sum(alpha_new**2))
    particle_treshold = float(float(config["N_PARTICLES"])/10.0)

    if N_eff > particle_treshold:
        x_current_all = x_new_all
        y_current_all = y_new_all
        theta_current_all = theta_new_all
        alpha_current = alpha_new

    else:
        all_indices = np.array([i for i in range(len(x_new_all))])
        new_indices = np.random.choice(all_indices, size = len(all_indices), p = alpha_new)
        x_current_all = x_new_all[new_indices]
        y_current_all = y_new_all[new_indices]
        theta_current_all = theta_new_all[new_indices]

        alpha_current = np.ones((config["N_PARTICLES"]))
        alpha_current = alpha_current/np.sum(alpha_current)
    
    x_best_all.append(x_best)
    y_best_all.append(y_best)
    theta_best_all.append(theta_best)
    
binary_map = get_binary_map(MAP['map'], 0.4)

plt.imshow(binary_map)
save_path = config["PLOTS_SAVE_FOLDER"] + str(config["DATASET"]) \
                    + "_map_final_particle_filter.png"
plt.savefig(save_path)
plt.show()
plt.pause(10)