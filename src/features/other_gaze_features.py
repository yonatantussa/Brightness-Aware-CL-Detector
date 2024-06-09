from glob import glob
from pathlib import Path
from tqdm import tqdm
from src.features.load_data import load_gaze_clipped
import cv2
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

def _fixation(points, grid_size, path_world_video):
    """Compute entropy based on the distribution of gaze points in a grid.

    This function calculates entropy using the distribution of gaze points in a grid.
    It quantizes gaze points into a grid and computes entropy based on their distribution.

    Parameters
    ----------
    points : array_like
        Array containing gaze points.
    grid_size : int
        Size of the grid used for quantizing gaze points.
    path_world_video : str or Path
        Path to the world video file.

    Returns
    -------
    float
        Entropy value calculated based on the distribution of gaze points in the grid.

    Notes
    -----
    This function quantizes gaze points into a grid defined by the grid_size and computes entropy
    based on the distribution of these points in the grid.
    """
    cap = cv2.VideoCapture(str(path_world_video))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    gaze = np.asarray(points)
    current_aoi = (0,0)

    all_durations = []
    current_duration = 0
    for gp in gaze:
        x_counter = np.floor(gp[0] / (width / grid_size))
        y_counter = np.floor(gp[1] / (height / grid_size))
        if (x_counter, y_counter) == current_aoi:
            current_duration += 1
        else:
            current_aoi = (x_counter, y_counter)
            if current_duration > 0:
                all_durations.append(current_duration)
            current_duration = 1
    all_durations.append(current_duration)

    mean_duration = np.mean(all_durations)
    return mean_duration


def fixation_duration(data_path, processed_step_path, grid_size=15, seconds=1):
    """Compute gaze entropy over time intervals for a given gaze data.

    This function calculates gaze entropy over time intervals for a specified gaze data
    and returns the calculated entropy values.

    Parameters
    ----------
    data_path : str or Path
        Path to the folder containing the gaze data and the world video file.
    processed_step_path : str or Path
        Path to the folder containing processed step data.
    grid_size : int, optional
        Size of the grid used for computing entropy. Default is 15.
    seconds : int, optional
        Length of time intervals in seconds for computing entropy. Default is 1.

    Returns
    -------
    list
        A list containing the computed gaze entropy values for each time interval.

    Notes
    -----
    This function processes gaze data and calculates entropy over specified time intervals.
    It utilizes multiprocessing to compute entropy in parallel for different segments of the gaze data.
    """
    # get world video path
    world_video = glob(str(Path(data_path) / '000' / 'world.mp4'))[0]

    # load the gaze file and estimate
    world_file = glob(str(Path(processed_step_path) / '000' / '*_DATA_WORLDTIME.json'))[0]
    cap = cv2.VideoCapture(str(world_video))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    gaze, world_time = load_gaze_clipped(world_file, width, height)

    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 5  # arbitrary default

     # Adjust the number of processes as needed
    fixation_durations = Parallel(n_jobs=cpus)(delayed(_fixation)(gaze[i:min(i + seconds * 30, len(gaze))], grid_size, world_video) for i in tqdm(range(0, len(gaze))))
    return fixation_durations

