#!/usr/bin/env python3
import json
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from pathlib import Path
import matplotlib.pyplot as plt
import os

__author__ = "Regine Bueter"
class PupilModelWithEstParams:
    """
    The Pupil Light Response (PLR) model estimates the pupil diameter based on the lightning situation. First it estimates the
    model parameters via a lest squares estimation.

    ...

    Attributes
    ----------
    a: float
        The parameter a for the pupil light response model
    b: float
        The parameter b for the pupil light response model
    delta: float
        The parameter delta  for the pupil light response model
    w1: float
        The parameter w1 for the pupil light response model. It estimates the influence the focal brightness on the
        users pupil response.
    w2: float
        The parameter a for the pupil light response model. It estimates the influence of the ambient brightness on the
        users pupil response.
    mse: float
        The mse of the signal of the parameter fit
    user: str
        The user

    Methods
    -------
    calculate_pupil_light_response(self, t, focal_b, ambient_b)
        Calculates the PLR (pupil diameter) from a given focal brightness and ambient brightness
    least_square_with_search_delta(self, pupil_diam, time_stamps, focal_brightness, ambient_light)
        Estimates the parameters for the PLR via a least squares approach with a grid search of delta
    save_parameters(self, save_path="models/Parameters.json"
        Saves the model parameters to a json file
    plot_estimation(self, pupil_diameter, time_stamps, focal_brightness, ambient_light, save_name,
                        save_path="reports/figures/", show_plot=False)
        Plots the estimated pupil signal and measured pupil signal for the estimated parameters

    """
    def __init__(self, user, load_parameters_from_file, path_to_params="models/Parameters.json"):
        """Constructor for the PLR model. Can either load the parameters from a file, or initialize the parameters with
        generic numbers

        Parameters
        ----------
        user: str
            The user for which the parameters are estimated for
        load_parameters_from_file: bool
            If the parameters should be loaded from a file
        path_to_params: str, optional:models/Parameters.json
            The path to the file, where the parameters are stored or should be stored
        """
        if load_parameters_from_file:
            data = json.load(open(path_to_params))[user]
            self.a = data["a"]
            self.b = data["b"]
            self.delta = data["delta"]
            self.w1 = data["w1"]
            self.w2 = data["w2"]
            self.mse = data["mse"]
            self.user = user
        else:
            self.user = user
            self.a = 0
            self.b = 0
            self.delta = 0
            self.w1 = 0.5
            self.w2 = 0.5
            self.mse = np.inf

    def calculate_pupil_light_response(self, t, focal_b, ambient_b):
        """Calculates the pupil light response from the focal and ambient brightness

        Parameters
        ----------
        t: array-like
            The timestamps to the focal and ambient brightness
        focal_b: array-like
            The focal brightness to estimate the pupil diameter on
        ambient_b: array-like
            The ambient brightness to estimate the pupil diameter on

        Returns
        -------
            array-like:
                The estimated pupil diameter
        """
        focal_b, ambient_b = self.__adapt_light(t, self.delta, focal_b, ambient_b)
        d = self.a * np.exp(self.b * (self.w1 * focal_b + self.w2 * ambient_b))
        return d

    def least_square_with_search_delta(self, pupil_diam, time_stamps, focal_brightness, ambient_light):
        """Estimates the parameters for the PLR model via a least squares estimation, with delta being estimated via
        grid search and minizining the mse

        Parameters
        ----------
        pupil_diam: array-like
            The measured pupil diameter to estimate the parameters on
        time_stamps: array-like
            The timestamps to the pupil diameter
        focal_brightness: array-like
            The according focal brightness
        ambient_light: array-like
            The according ambient brightness
        """
        # optimize for RMSE
        idx_bw_image = int(list(map(lambda i: i > 30, time_stamps)).index(True))
        if ambient_light[idx_bw_image + 10] != 255:
            slope_data = True
        else:
            slope_data = False
        mse_opt = np.inf
        for d in np.arange(-30, 0, 0.1):
            t = time_stamps
            i = self.__adapt_light(t, d, focal_brightness, ambient_light)
            a_0 = np.max(pupil_diam)
            b_0 = -0.01
            w1_0 = 0.5
            w2_0 = 0.5
            popt, pcov = curve_fit(self.__exp_model, xdata=i, ydata=pupil_diam,
                                   bounds=([0.0000001, -np.inf, 0, 0], [10, 0, 1, 1]),
                                   p0=(a_0, b_0, w1_0, w2_0))

            mse = mean_squared_error(pupil_diam, self.__exp_model(i, *popt))
            if mse < mse_opt:
                mse_opt = mse
                self.a = popt[0]
                self.b = popt[1]
                self.w1 = popt[2]
                self.w2 = popt[3]
                self.delta = d
        self.mse = mse_opt

    def save_parameters(self, save_path="models/Parameters.json"):
        """Saves the internal estimated parameters to a file

        Parameters
        ----------
        save_path: str
            The json file to save the parameters to

        """
        if not Path(save_path).parent.exists():
            os.mkdir(Path(save_path).parent)
        if not Path(save_path).exists():
            f = open(save_path, 'x')
            output = {}
        else:
            f = open(save_path)
            output = json.load(f)

        output[self.user] = {"a": self.a, "b": self.b, "delta": self.delta, "w1": self.w1, "w2": self.w2,
                             "mse": self.mse}

        out_file = open(save_path, "w")
        json.dump(output, out_file, indent=6)
        out_file.close()

    def plot_estimation(self, pupil_diameter, time_stamps, focal_brightness, ambient_light, save_name,
                        save_path="reports/figures/", show_plot=False):
        """Plots the estimation of the measured pupil diameter and estimates the pupil diameter based on the focal
        brightness and ambient brightness and plots it, as well as both brightnesses. It can be saved to the file
        system, or just be shown.

        Parameters
        ----------
        pupil_diam: array-like
            The measured pupil diameter to estimate the parameters on
        time_stamps: array-like
            The timestamps to the pupil diameter
        focal_brightness: array-like
            The according focal brightness
        ambient_light: array-like
            The according ambient brightness
        save_name: str
            The name of the plot, that it should be named on the file system
        save_path: str, optional:reports/figures/
            The path to save the plot to
        show_plot: bool, default:False
            If true the plot will be shown.

        """
        i = self.__adapt_light(time_stamps, self.delta, focal_brightness, ambient_light)

        fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
        ax.plot(time_stamps, pupil_diameter, color='b', label="pupil_diam")
        ax.plot(time_stamps, self.__exp_model(i, self.a, self.b, self.w1, self.w2), color='orange',
                label='fit: a=%5.3f, b=%5.3f, delta=%5.3f, w1=%5.3f, w2=%5.3f' % tuple(
                    np.array([self.a, self.b, self.delta, self.w1, self.w2])))
        ax.plot(time_stamps, focal_brightness / 255, color='red', label="focal_brightness normalized [0,1]")
        ax.plot(time_stamps, ambient_light / 255, color='green', label="ambient brightness normalized [0,1]")
        ax.grid()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Diameter (mm)')

        fig.legend(loc='outside lower center')
        save_path = Path(save_path) / self.user
        if not save_path.exists():
            os.makedirs(save_path)
        plt.savefig(f"{save_path}/{save_name}.png", dpi=300)
        if show_plot:
            plt.show()
        else:
            plt.close()

    def __exp_model(self, i, a, b, w1, w2):
        """Estimates the pupil diameter based on the given parameters and the intensity i (focal brightness and ambient
        brightness)

        Parameters
        ----------
        i: tuple
            A tuple consisting of the array of the focal brightness and ambient brightness
        a: float
            The parameter a
        b: float
            The parameter b
        w1: float
            The parameter w1
        w2: float
            The parameter w2

        Returns
        -------
            array-like:
                The pupil diameter
        """
        fb, ab = i
        d = a * np.exp(b * (w1 * fb + w2 * ab))
        return d

    def __adapt_light(self, t, d, focal_b, ambient_b):
        """Adapts the focal and ambient brightness to delta, to fit the according indices to the PLR model

        Parameters
        ----------
        t: array-like
            The timestamps for the focal and ambient brightness
        d: float
            The parameter delta of the PLR model, delta is given in seconds.
        focal_b: array-like
            The focal brightness to adapt to the delta
        ambient_b: array-like
            The ambient brightness to adapt to delta

        Returns
        -------
         array-like:
            The adapted focal brightness
        array-like:
            The adapted ambient brightness
        """
        if not isinstance(t[0], float):
            t = np.array(t).astype(np.float)
        adapted_t = np.add(t, d)
        if adapted_t[0] >= 0:
            idx = (np.abs(t - adapted_t[0])).argmin()
            focal_b = np.concatenate((focal_b[idx:], np.full(idx, focal_b[-1])))
            ambient_b = np.concatenate((ambient_b[idx:], np.full(idx, ambient_b[-1])))
        else:
            num_idx_adding_neg = (np.abs(t - np.abs(adapted_t[0]))).argmin()
            focal_b = np.concatenate(
                (np.full(num_idx_adding_neg, focal_b[0]), focal_b[: len(focal_b) - num_idx_adding_neg]))
            ambient_b = np.concatenate(
                (np.full(num_idx_adding_neg, ambient_b[0]), ambient_b[: len(ambient_b) - num_idx_adding_neg]))
        return focal_b, ambient_b
