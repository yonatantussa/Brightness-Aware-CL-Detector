#!/usr/bin/env python3
from pathlib import Path
import argparse
import os
import json
import math
from src.features.Pupil_Model import PupilModelWithEstParams
from src.features.load_data import load_concatenated_data_filtered
from sklearn.metrics import mean_squared_error

__author__ = "Regine Bueter"


def calculate_prediction_error(args, path_user):
    """Calculates the prediction error (mse and rmse) on step 8 of the Experiment procedure (Color Calibration 3)

    Parameters
    ----------
    args: ArgumentParser
        The arguments from the code
    path_user: str
        The path to the user, to load the files

    """
    # load data
    use_step = [args.prediction_error_step]
    pupil_diam, time_stamps, focal_brightness, ambient_light, cognit_load = load_concatenated_data_filtered(path_user,
                                                                                                            use_step)
    # get Pupil Light Response Model
    pupil_model = PupilModelWithEstParams(user=args.user, load_parameters_from_file=True)
    # calculate prediction diameter
    est_diam = pupil_model.calculate_pupil_light_response(time_stamps, focal_brightness, ambient_light)
    # calculate mse and rmse between prediction and ground truth
    mse = mean_squared_error(pupil_diam, est_diam)

    # print results
    print(f"[RESULT] MSE of prediction: {mse}")
    print(f"[RESULT] RMSE of prediction: {math.sqrt(mse)}")

    # save results
    save_path = Path(f"{args.save_path}/results/") / "mse_prediction_CC_3.json"
    if not Path(save_path).parent.exists():
        os.mkdir(Path(save_path).parent)
    if not Path(save_path).exists():
        f = open(save_path, 'x')
        output = {}
    else:
        f = open(save_path)
        output = json.load(f)

    output[args.user] = {"mse": mse}

    out_file = open(save_path, "w")
    json.dump(output, out_file, indent=6)
    out_file.close()


def estimate_parameters(args):
    """Estimate parameters for the Pupil Light Response Model based on provided arguments.

    This function performs parameter estimation using the Pupil Light Response Model.
    It utilizes the least square with search delta method to estimate parameters
    and plot the estimation based on the provided data and settings.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments.

    Returns
    -------
    None

    Notes
    -----
    This function is responsible for estimating parameters using the Pupil Light Response Model
    based on the provided data and settings. It utilizes the least square with search delta method
    and prints out the results including estimated parameters, mean square error (MSE), and root
    mean square error (RMSE) of the fit.

    Example
    -------
    >>> estimate_parameters(parsed_arguments)
    # Output: Performs parameter estimation using the Pupil Light Response Model
    # based on the provided arguments.
    """
    print(f"----------------------- {args.user} -----------------------")
    print(f"[INFO] steps used: {args.steps}")
    # get Pupil Light Response Model
    pupil_model = PupilModelWithEstParams(user=args.user, load_parameters_from_file=False)

    # load data
    path = str(Path(args.folder_path) / args.eye / args.user)
    pupil_diam, time_stamps, focal_brightness, ambient_light, cognit_load = load_concatenated_data_filtered(path, args.steps)

    # estimate parameters
    pupil_model.least_square_with_search_delta(pupil_diam, time_stamps, focal_brightness, ambient_light)
    pupil_model.save_parameters()
    pupil_model.plot_estimation(pupil_diam, time_stamps, focal_brightness, ambient_light, save_name='least_square_training')

    # print results of the least square estimation
    print(f"[RESULT] Estimated parameters: "
          f"a: {pupil_model.a}, b: {pupil_model.b}, delta: {pupil_model.delta}, "
          f"w1: {pupil_model.w1}, w2: {pupil_model.w2}")
    print(f"[RESULT] MSE of fit: {pupil_model.mse}")
    print(f"[RESULT] RMSE of fit: {math.sqrt(pupil_model.mse)}")

    # calculate prediction error
    calculate_prediction_error(args, path)

def main(args):
    """Estimate parameters for users and steps based on the provided trial file.

    This function iterates through each user and their respective steps based on the given
    folder path and eye type. It loads the trial file to define the steps and users to be used
    for parameter estimation. If a user and its step combination are found in the trial file,
    it proceeds to estimate parameters using the `estimate_parameters` function.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments.

    Raises
    ------
    FileNotFoundError
        If the trial file is not found.

    Notes
    -----
    This function is responsible for estimating parameters for users and their steps based on
    the trial file. It identifies the steps and users to be used for parameter estimation,
    calling the `estimate_parameters` function for the estimation process.

    Example
    -------
    >>> main(parsed_arguments)
    # Output: Performs parameter estimation based on the provided trial file and arguments.
    """
    # if it cannot find the trial file, raise an error,
    # otherwise load the trial file and define the steps and users to be used
    if not Path(args.trial).exists():
        raise FileNotFoundError
    with open(args.trial) as file:
        use_data = [line.rstrip() for line in file if len(line.rstrip()) != 0]

    # iterate through all users and steps and estimate the parameters, if the user and step is in the trial file
    all_users_path = os.path.join(args.folder_path, args.eye)
    all_users = sorted(os.listdir(all_users_path))
    for user in all_users:
        if user == ".DS_Store" or "._" in user:
            continue
        args.user = user
        # get all steps of the user
        all_steps_path = os.path.join(all_users_path, user)
        all_steps = os.listdir(all_steps_path)

        # get the steps that should be used for the estimation
        estimating_steps = []
        for step in all_steps:
            if f"{user}:{step}" in use_data:
                estimating_steps.append(step)

        # if there are steps to be estimated, estimate the parameters
        if len(estimating_steps) != 0:
            args.steps = estimating_steps
            estimate_parameters(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for Probabilistic Model")
    parser.add_argument("--folder_path", required=False, type=str, default="data/processed",
                        help="Path to the data folder containing the processed data")
    parser.add_argument("--prediction_error_step", type=str, default="step-8-pupil-calibration-4",
                        help="The step for which the prediction error should be calculated")
    parser.add_argument("--eye", type=str, default="right", choices=["right", "left"],
                        help="The eye to be used")
    parser.add_argument("--trial", type=str, default='from_file',
                        help='The usable trials can be loaded from a file, if a valid path to a txt file is given or '
                             'the default from the new_data and phantom data can be used. The txt file should be in '
                             'the format user:experiment_step in each line')
    parser.add_argument("--save_path", type=str, default='reports',
                        help="Path to save the results and figures")
    args = parser.parse_args()

    main(args)
