#!/usr/bin/env python3
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy.stats import norm
from src.features.Pupil_Model import PupilModelWithEstParams
from src.features.load_data import load_concatenated_data_filtered


__author__ = "Paola Ruiz Puentes, Regine Bueter"


def normal_dist_PD_FB(x, mean, sd):
    """Calculates the Probability value for cognitive load

    Parameters
    ----------
    x: float
        The pupil diameter
    mean: float
        The mean for the probabilistic model
    sd: float
        The standard deviation

    Returns
    -------
        float: The probability for cognitive load
    """
    prob_val = norm.cdf(x, mean, sd)
    return prob_val


def calculate_standard_deviation(path_user, eye):
    """Calculates the standard deviation for the probabilistic model. It is based on the standard deviation of the
    maximum pupil diameters (during the black image)

    Parameters
    ----------
    path_user: str
        The path to the folder with the data
    eye: str
        The eye to be analyzed

    Returns
    -------
        float: The standard deviation for the probabilistic model
    """
    if eye == "right":
        eye_id = 0
    elif eye == "left":
        eye_id = 1
    else:
        raise ValueError("Eye must be either 'right' or 'left'")

    # get path for first step
    all_steps = os.listdir(path_user)
    step_1 = [k for k in all_steps if "step-0" in k and '._' not in k][0]

    df = pd.read_csv(Path(path_user) / step_1 / "pupil_positions.csv")
    diameter = (df.diameter_3d[(df.method == "pye3d 0.3.0 real-time") & (df.eye_id == eye_id)].to_numpy().astype(float))
    if 'slope' in step_1:
        # load timestamps
        try:
            time_stamps = (
                df.pupil_timestamp[(df.method == "pye3d 0.3.0 real-time") & (df.eye_id == eye_id)].to_numpy().astype(
                    float))
        except:
            time_stamps_tmp = df.pupil_timestamp[
                (df.method == "pye3d 0.3.0 real-time") & (df.eye_id == eye_id)].to_list()
            time_stamps = []
            for time in time_stamps_tmp:
                time_stamps.append("".join(c for c in time if (c.isdigit() or c == ".")))
            time_stamps = np.array(time_stamps).astype(float)
        # get idx where black image ends
        idx_black_ends = int(list(map(lambda i: i > 30, time_stamps)).index(True))
        return np.std(diameter[0:idx_black_ends])
    else:
        return np.std(diameter)


def estimate_probability(path, user, steps, eye, save_path):
    """Estimates the probability for cognitive load based on the probabilistic model

    Parameters
    ----------
    path: str
        The path to the folder with the data for the user
    user: str
        The user to be analyzed
    steps: list
        The steps to be analyzed
    eye: str
        The eye to be analyzed
    save_path: str
        The path to save the figures

    Returns
    -------
        np.array: The probability values for cognitive load
        np.array: The cognitive load values
    """
    global focal_brightness
    global ambient_light

    # get the correct order of all steps
    pupil_model = PupilModelWithEstParams(user=user, load_parameters_from_file=True)
    pupil_diam, time_stamps, focal_brightness, ambient_light, cognit_load = load_concatenated_data_filtered(path, steps)
    est_diam = pupil_model.calculate_pupil_light_response(time_stamps, focal_brightness, ambient_light)

    std = calculate_standard_deviation(path, eye)

    # * Evaluating CL signal on data.
    probability = []
    # for i, diam in enumerate(tqdm(pupil_diam)):
    for i, diam in enumerate(pupil_diam):
        # estimate probabilities
        prob = normal_dist_PD_FB(diam, mean=est_diam[i], sd=std)
        probability.append(prob)

    # get np arrays from list
    probability = np.asarray(probability)

    return probability, cognit_load


def probablistic_model(args):
    """Estimate probabilities based on the specified steps and user.

    This function loads the necessary data and estimates probabilities for the given user
    and steps. It then calculates the probability and cognitive load for the specified user
    and saves the results.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments.

    Returns
    -------
    tuple
        A tuple containing the estimated probability and cognitive load.

    Notes
    -----
    This function estimates probabilities for the specified user based on the provided steps.
    It calculates and returns the probability and cognitive load derived from the input data.

    Example
    -------
    >>> result_probability, result_cognit_load = probablistic_model(parsed_arguments)
    # Output: Prints information about the user and steps used, and returns estimated probability and cognitive load.
    """

    # get the correct order of all steps
    path = str(Path(args.folder_path) / args.eye / args.user)
    probability, cognit_load = estimate_probability(path, args.user, args.steps, args.eye, args.save_path)

    return probability, cognit_load


def main(args):
    """Estimate probabilities and perform statistical analysis based on provided arguments.

    This function loads the necessary trial data, iterates through users and steps,
    and estimates probabilities using the `probablistic_model` function. It performs statistical
    analysis on the estimated probabilities based on the provided arguments.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the specified trial file is not found.

    Notes
    -----
    This function is responsible for iterating through all users and steps, estimating
    probabilities, and performing statistical analysis based on the provided command-line arguments.

    Example
    -------
    >>> main(parsed_arguments)
    # Output: Performs probability estimation and statistical analysis based on the provided arguments.
    """
    if not Path(args.trial).exists():
        raise FileNotFoundError
    with open(args.trial) as file:
        use_data = [line.rstrip() for line in file if len(line.rstrip()) != 0]

    # iterate through all users and steps and estimate the parameters, if the user and step is in the trial file
    all_users_path = os.path.join(args.folder_path, args.eye)
    all_users = sorted(os.listdir(all_users_path))
    for user in all_users:
        if user == ".DS_Store" or '._' in user:
            continue
        args.user = user

        # get correct steps for current user
        all_steps_path = os.path.join(all_users_path, user)
        all_steps = os.listdir(all_steps_path)
        estimating_steps = []
        for step in all_steps:
            if f"{user}:{step}" in use_data:
                estimating_steps.append(step)

        if len(estimating_steps) != 0:
            # if individual steps should be used for analysis, iterate through all steps
            if args.individual_steps:
                for step in estimating_steps:
                    # define correct step for analysis
                    args.steps = [step]
                    # estimate probability
                    probability, cognit_load = probablistic_model(args)
                    # add values to statistical analysis
            else:
                # if all steps should be used for analysis, use all steps
                args.steps = estimating_steps
                # estimate probability
                probability, cognit_load = probablistic_model(args)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for Probabilistic Model")
    parser.add_argument("--folder_path", required=False, type=str, default="data/processed",
                        help="Path to the folder of the users data")
    parser.add_argument("--individual_steps", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="If true, the steps are analyzed individually")
    parser.add_argument("--eye", type=str, default="right",
                        help="The eye that should be used for analysis")
    parser.add_argument("--trial", type=str,
                        help='The usable trials can be loaded from a file, if a valid path to a txt file is given or '
                             'the default from the new_data and phantom data can be used. The txt file should be in '
                             'the format user:experiment_step in each line')
    parser.add_argument("--save_path", type=str, default='reports',
                        help='Path to save the figures')
    args = parser.parse_args()

    main(args)
