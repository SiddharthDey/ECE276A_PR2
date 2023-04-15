All the python files can be run directly once "DATA_FOLDER" in the configPR2.yaml is changes to the current system's data folder location

The folder contains the following python codes:

- first_lidar_scan_mapping.py: code to generate the binary map using the first lidar scan - Q1)a) Mapping
- dead_reckoning_trajectory.py: code to generate the trajectory for dead reckoning
- dead_reckoning_mapping.py: code to generate the map using the dead-reckoning trajectory
- particle_filter_trajectory_prediction.py: code to calculate the trajectory of N particles (default: N = 100)
- particle_filter_mapping.py: code to generate the map using the particle filter prediction and update algorithm with resampling
- texture_map.py: code to generate the texture map (needs the best_trajectory_[dataset].npy file to load the best trajectory for global location)

- configPR2.yaml: a YAML file imported in all the codes, used to change the hyperparameters of the codes as mentioned below



The folder contains the following two sub-folders:
- plots/ : folder containing all the plots, all the codes save the plots in this folder
- results/ : folder containing the best trajectory npy files for both datasets, is used by texture_map.py to generate the texture map
- videos/ : The videos of occupancy grip mad updates and texture map updates with time


The configPR2.yaml file contains the following hyperparameters which can be changed:
- DATA_FOLDER: (only need to change this to run all the codes, other hyperparameters are optional to change) the path to the data folder

- PLOTS_SAVE_FOLDER: the path to folder where all the plots are saved
- RESULTS_FOLDER: the path to folder where the best trajectory npy file generated by particle_filter_mapping.py is saved
- MAP_X_MIN: minimum x value of the map in meters
- MAP_Y_MIN: minimum y value of the map in meters
- MAP_X_MAX: maximum x value of the map in meters
- MAP_Y_MAX: maximum y value of the map in meters
- MAP_RES: resolution of the map in meters
- N_PARTICLES: number of particles to consider for particle filter algorithm
- SIGMA_V: standard deviation of the noise added to linear velocity
- SIGMA_W: standard deviation of the noise added to angular velocity
- FLOOR_Z_RANGE: hyperparameter to filter out the points belonging to the floor in texture_map.py, the points with z belonging to       (-FLOOR_Z_RANGE, +FLOOR_Z_RANGE) are considered for color mapping


The videos folder contains 4 videos:
- 20_occupancy_map.avi: occupancy grid map updates with time for dataset 20
- 21_occupancy_map.avi: occupancy grid map updates with time for dataset 21
- 20_texture_map.avi: texture grid map updates with time for dataset 20
- 21_texture_map.avi: texture grid map updates with time for dataset 21 
