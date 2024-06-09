#!/usr/bin/env python3
import numpy as np
import json
import pandas as pd
from src.features.process_data import mean_filter_fb, mean_filter_signal
from pathlib import Path
import os

__author__ = "Paola Ruiz Puentes, Regine Bueter"

def load_json_signal(path_to_file):
    """
    Loads pupillometry information.
    PUPILTIME file has 30hz data
    WORLDTIME file has 120hz data

    Parameters
    ----------
    path_to_file: str
        Path to the pupillometry file.

    Returns
    -------
    d: array_like
        Pupil diameter
    d_t: array_like
        Timestamps
    d_cl: array_like
        Cognitive load labels
    d_fb: array_like
        Focal brightness: computed over a 11x11 pixels grid, centered on the gaze point.
    d_ab: array_like
        Ambient brightness: computed as the mean of the image in gray.
    pd_d: pd.DataFrame
        Pandas dataframe with all the information above.
    """
    f = open(path_to_file)
    pupil_signal = json.load(f)
    if "PUPILTIME" in path_to_file:

        d = pupil_signal["pupil_diameter"]
        d_t = pupil_signal["base_time"]
        d_cl = pupil_signal["cognitive_load"]
        d_fb = pupil_signal["focal_brightness"]
        d_ab = pupil_signal["ambient_brightness_1d"]

        pd_diameter = pd.DataFrame.from_dict(pupil_signal)

    elif "WORLDTIME" in path_to_file:

        d_t = pupil_signal["world_time"]
        d_cl = pupil_signal["cognitive_load"]
        d = pupil_signal["pupil_diameter"]
        d_fb = pupil_signal["focal_brightness"]
        d_ab = pupil_signal["ambient_brightness_1d"]

        pd_diameter = pd.DataFrame.from_dict(pupil_signal)
    else:
        raise FileNotFoundError("Path to file was given wrong, must be either \"PUPILTIME\" or \"WORLDTIME\" ")
    d_cl = np.asarray(d_cl)
    d_cl[d_cl > 0] = d_cl[d_cl > 0] - 1
    d_cl = d_cl.tolist()
    return d, d_t, d_cl, d_fb, d_ab, pd_diameter


def load_csv_data(path):
    """
    Loads pupil diameter and timestamps from pupil_positions.csv file

    Parameters
    ----------
    path: str
        Path to csv file

    Returns
    -------
    diameter: array_like
        Pupil diameter
    time_stamps: array_like
        Timestamps
    """
    eyeID = 0
    df = pd.read_csv(path)
    diameter = df.diameter_3d[(df.method == "pye3d 0.3.0 real-time") & (df.eye_id == eyeID)].to_numpy().astype(float)
    try:
        time_stamps = df.pupil_timestamp[
            (df.method == "pye3d 0.3.0 real-time") & (df.eye_id == eyeID)].to_numpy().astype(float)
    except:
        time_stamps_tmp = df.pupil_timestamp[(df.method == "pye3d 0.3.0 real-time") & (df.eye_id == eyeID)].to_list()
        time_stamps = []
        for time in time_stamps_tmp:
            time_stamps.append(''.join(c for c in time if (c.isdigit() or c == '.')))
        time_stamps = np.array(time_stamps).astype(float)
    return diameter, time_stamps


def load_step_timestamps_info(args):
    path_file = Path("pupilcapture_times") / args.user / args.step / "_events_pupil_times.json"
    f = open(path_file)
    exps_times_info = json.load(f)
    del_keys = [key for key in exps_times_info.keys() if "expID" in key]
    for del_key in del_keys:
        del exps_times_info[del_key]

    return exps_times_info


def append_light(signal, light, mode, step=None):
    """
    Sets the ambient light for a given signal with a given mode

    Parameters
    ----------
    signal: array_like
        The signal to use for the estimate the ambient brightness, can either be the time_stamps for the bw image or the
        slope or the ambient brightness signal itself for the rest
    light: array_like
        The output array to concatenate the new ambient light
    mode: {'bw-image', 'slope', 'brightness'}
        The mode for which the brightness should be calculated, can either be 'bw-image' for black and white image, or
        'slope' for the slope data or 'brightness' for data, that uses the ambient brightness from the json files
    step: optional, {'black', 'white'}
        Needs to be provided for the black and white image calculation can either be 'black' or 'white'

    Returns
    -------
    array_like
        The ambient light signal
    """
    # black and white image
    if mode == "bw-image":
        if "black" in step:
            light = np.concatenate((light, np.full(len(signal), 0.0)))
        else:
            light = np.concatenate((light, np.full(len(signal), 255)))

    # slope data
    elif mode == "slope":
        # signal is time_stamps
        # find first index, where time_stamps reaches 30, etc
        idx_black_ends = int(list(map(lambda i: i > 30, signal)).index(True))
        idx_slope_starts = idx_black_ends
        idx_slope_ends = int(list(map(lambda i: i > 90, signal)).index(True))
        idx_white_starts = idx_slope_ends

        light = np.zeros_like(signal)
        light[:idx_black_ends].fill(0)
        num_steps_in_min = 60
        idx_start = idx_slope_starts
        increasing_val = 255 / num_steps_in_min
        for i in range(num_steps_in_min):
            idx_end = int(list(map(lambda x: x > 31 + i, signal)).index(True))
            light[idx_start:idx_end] = increasing_val * i
            idx_start = idx_end
        light[idx_white_starts:].fill(255)

    # use focal brightness signal
    else:
        # fb data as signal
        light = np.concatenate((light, np.array(signal)))
    return light


def sort_steps(steps: list):
    """Sorts the given steps according to their order conduced in the study

    Parameters
    ----------
    steps: list
        The list of steps, that need to be sorted

    Returns
    -------
        list: The given steps, but sorted
    """
    steps_sorted = []
    step_counter = True
    for step in steps:
        if 'step' not in step:
            step_counter = False

    if step_counter:
        for idx, step in enumerate(steps):
            count_str = step[:8]
            k = sum(list(map(lambda x: 1 if x.isdigit() else 0, set(count_str))))
            if k <= 1:
                step = step[:5] + "0" + step[5:]
                steps[idx] = step
        steps = np.sort(steps)
        # throw out the zero that was inserted for the sorting
        for step in steps:
            if step[5] == '0':
                steps_sorted.append(step[:5] + step[6:])
            else:
                steps_sorted.append(step)
    else:
        for idx, step in enumerate(steps):
            if 'black' in step:
                steps[idx] = '01_' + step
            elif 'white' in step:
                steps[idx] = '02_' + step
            elif 'Color' in step:
                num = int(step[-1]) * 2 + 1
                if num > 9:
                    steps[idx] = f'{int(num)}_' + step
                else:
                    steps[idx] = f'0{int(num)}_' + step
            elif 'Peg' in step:
                num = int(step[-1]) * 2 + 2
                if num > 9:
                    steps[idx] = f'{int(num)}_' + step
                else:
                    steps[idx] = f'0{int(num)}_' + step
        steps = np.sort(steps)
        for step in steps:
            steps_sorted.append(step[3:])
    return steps_sorted


def load_concatenated_data_filtered(path: str, steps: list, extract_first_chatter=False, exp_id='000',
                                    filter_type='mean', kernel_size=1443):
    """
    Loads the data from the given path to a user for the given steps and filters them

    Parameters
    ----------
    path: str
        The path to the folder of the user
    steps: list
        The list of steps to use
    extract_first_chatter: bool, default=False:
        If concatenated signal is given, the first chatter will only be used
    exp_id:str, default='000'
        The exportID number
    filter_type:{'mean', 'median'}
        Specifies the type of filter to be used
    kernel_size: int, default=1443
        Defines the kernel_size used for the butterworth filter in number of samples

    Returns
    -------
    pupil_diam: array_like
        The pupil diameter after filtering
    time_stamps: array_like
        The respective time stamps
    focal_brightness: array_like
        The focal brightness after filtering
    ambient_light: array_like
        The ambient light after filtering
    """
    focal_brightness = np.array([])
    ambient_light = np.array([])
    pupil_diam = np.array([])
    time_stamps = np.array([])
    cognit_load = np.array([])

    # sort steps to get correct order
    steps = sort_steps(steps)

    # load each step
    for step in steps:
        if '.DS_Store' == step:
            continue

        if "black-image" in step or "white-image" in step:
            mode = "bw-image"
            try:
                data_path = f"{path}/{step}/pupil_positions.csv"
                pupil_diam_local, time_stamps_local = load_csv_data(data_path)
            except:
                data_path = f"{path}/{step}/{exp_id}/pupil_positions.csv"
                pupil_diam_local, time_stamps_local = load_csv_data(data_path)
            focal_brightness = append_light(time_stamps_local, focal_brightness, mode, step)
            ambient_light = append_light(time_stamps_local, ambient_light, mode, step)
            cognit_load_local = np.zeros_like(pupil_diam_local)

        elif "slope" in step:
            mode = "slope"
            data_path = f"{path}/{step}/pupil_positions.csv"
            pupil_diam_local, time_stamps_local = load_csv_data(data_path)
            focal_brightness = append_light(time_stamps_local, focal_brightness, mode)
            ambient_light = append_light(time_stamps_local, ambient_light, mode)
            cognit_load_local = np.zeros_like(pupil_diam_local)

        else:
            mode = "brightness"
            data_path = f"{path}/{step}/{exp_id}/{step}_{exp_id}_DATA_WORLDTIME.json"
            pupil_diam_local, time_stamps_local, cognit_load_local, focal_b, ambient_b, all_data_diameter = load_json_signal(
                data_path)
            if extract_first_chatter and "endback" in step:
                chatter_end_idx = np.argmax(np.diff(cognit_load_local)) + 1
                pupil_diam_local = pupil_diam_local[:chatter_end_idx]
                time_stamps_local = time_stamps_local[:chatter_end_idx]
                focal_b = focal_b[:chatter_end_idx]
                ambient_b = ambient_b[:chatter_end_idx]
            fb_filtering = mean_filter_fb(focal_b, filter_type, kernel_size)
            focal_brightness = append_light(fb_filtering, focal_brightness, mode)
            ab_filtering = mean_filter_fb(ambient_b, filter_type, kernel_size)
            ambient_light = append_light(ab_filtering, ambient_light, mode)

        cognit_load = np.concatenate((cognit_load, cognit_load_local))
        pupil_diam = np.concatenate((pupil_diam, pupil_diam_local))
        if len(time_stamps) != 0:
            time_stamps_local = np.add(time_stamps_local, time_stamps[-1])
        time_stamps = np.concatenate((time_stamps, time_stamps_local))

    pupil_diam = mean_filter_signal(pupil_diam, filter_type, kernel_size)

    return pupil_diam, time_stamps, focal_brightness, ambient_light, cognit_load


def load_step_filtered(path_user: str, step: str, exp_id='000', filter_type='mean', kernel_size=1443):
    """
    Loads one step from given user path filters them

    Parameters
    ----------
    path_user: str
        The path to the folder of the user
    step: str
        The step to load
    exp_id:str, default='000'
        The exportID number
    filter_type:{'mean', 'median'}
        Specifies the type of filter to be used
    kernel_size: int, default=1443
        Defines the kernel_size used for the butterworth filter in number of samples

    Returns
    -------
    pupil_diam: array_like
        The pupil diameter after filtering
    time_stamps: array_like
        The respective time stamps
    focal_brightness: array_like
        The focal brightness after filtering
    ambient_light: array_like
        The ambient light after filtering
    """
    time_offset = 0
    all_steps = os.listdir(path_user)
    all_steps = [step for step in all_steps if '._' not in step]

    steps = sort_steps(all_steps)
    # throw out the zero that was inserted for the sorting
    for s in steps:
        if s == ".DS_Store":
            continue
        if s == step:
            break
        if "black-image" in s or "white-image" in s or 'slope' in s:
            try:
                data_path = f"{path_user}/{s}/pupil_positions.csv"
                _, time_stamps_local = load_csv_data(data_path)
            except:
                data_path = f"{path_user}/{s}/{exp_id}/pupil_positions.csv"
                _, time_stamps_local = load_csv_data(data_path)
        else:
            data_path = f"{path_user}/{s}/{exp_id}/{s}_{exp_id}_DATA_WORLDTIME.json"
            _, time_stamps_local, _, _, _, _ = load_json_signal(data_path)
        time_offset = + time_stamps_local[-1]

    if type(step) != list:
        step = [step]
    pupil_diam, time_stamps, focal_brightness, ambient_light, cognit_load = load_concatenated_data_filtered(path_user,
                                                                                                            step,
                                                                                                            exp_id=exp_id,
                                                                                                            filter_type=filter_type,
                                                                                                            kernel_size=kernel_size)
    return pupil_diam, time_stamps + time_offset, focal_brightness, ambient_light, cognit_load


def load_steps_of_user_into_dict(path_user: str, steps: list, exp_id='000', max_n_back_task=3):
    """
    Loads the data from the given path to a user for the given steps into a dictionary, where they are sorted by the
    performed n-back task. If several sequences of a cognitive load value are present, they are appended in a
    separate list.

    Parameters
    ----------
    path_user: str
        The path to the folder of the user
    steps: list
        The list of steps to use
    exp_id:str, default='000'
        The exportID number
    max_n_back_task: int, default=2
        The maximum perfomed n-back task

    Returns
    -------
        dict: A dict of n-back tasks from the given data of a user with the given steps,
              one sequence contains the pupil diameter, the cognitive load labels and the timestamps
    """
    dict_sequences = {str(x): [] for x in range(0, max_n_back_task + 1)}

    steps = sort_steps(steps)
    # throw out the zero that was inserted for the sorting
    for step in steps:
        if '.DS_Store' == step:
            continue

        data_path = f"{path_user}/{step}/{exp_id}/{step}_{exp_id}_DATA_WORLDTIME.json"
        pupil_diam_local, time_stamps_local, cognit_load_local, focal_b, ambient_b, all_data_diameter = load_json_signal(
            data_path)
        cognit_load_local = np.asarray(cognit_load_local)

        prev_cl = -1
        bottom_idx = 0
        for idx, coag_load in enumerate(cognit_load_local):
            current_cl = coag_load
            if idx == 0:
                prev_cl = current_cl
                continue
            if idx == len(pupil_diam_local) - 1:
                d_t = time_stamps_local[bottom_idx:]
                d_cl = cognit_load_local[bottom_idx:]
                d = pupil_diam_local[bottom_idx:]

                information_segment = [d, d_cl, d_t]
                dict_sequences[str(int(prev_cl))].append(information_segment)

            if current_cl != prev_cl:
                top_idx = idx

                d_t = time_stamps_local[bottom_idx:top_idx]
                d_cl = cognit_load_local[bottom_idx:top_idx]
                d = pupil_diam_local[bottom_idx:top_idx]

                information_segment = [d, d_cl, d_t]
                dict_sequences[str(int(prev_cl))].append(information_segment)

                bottom_idx = top_idx
                prev_cl = current_cl

    return dict_sequences

def load_gaze(path_to_file):
    """
        Loads gaze information.
        PUPILTIME file has 30hz data
        WORLDTIME file has 120hz data

        Parameters
        ----------
        path_to_file: str
            Path to the pupillometry file.

        Returns
        -------
        g: array_like
            gaze_points [x,y]
        d_t: array_like
            Timestamps
        """
    f = open(path_to_file)
    pupil_signal = json.load(f)
    if "WORLDTIME" in path_to_file:

        g = pupil_signal['gaze_xy']
        d_t = pupil_signal["world_time"]

    elif "PUPILTIME" in path_to_file:

        d_t = pupil_signal["base_time"]
        g = pupil_signal['gaze_xy']

    else:
        raise FileNotFoundError("Path to file was given wrong, must be either \"PUPILTIME\" or \"WORLDTIME\" ")
    return g, d_t


def load_gaze_clipped(path_to_file, width, height):
    """
    Loads gaze information clipped to the video width and height
    PUPILTIME file has 30hz data
    WORLDTIME file has 120hz data

    Parameters
    ----------
    path_to_file: str
        Path to the pupillometry file.
    width: int
        Width of the video
    height: int
        Height of the video

    Returns
    -------
    g: array_like
        gaze_points [x,y]
    d_t: array_like
        Timestamps
    """
    g, d_t = load_gaze(path_to_file)

    gaze = np.asarray(g)
    gaze_x = np.clip(gaze[:, 0], 0, width)
    gaze_y = np.clip(gaze[:, 1], 0, height)

    return np.asarray([gaze_x, gaze_y]).T, d_t

