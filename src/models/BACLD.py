import argparse
import os
from glob import glob
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import sklearn
import pickle
import shap
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, confusion_matrix, make_scorer

from src.features.load_data import load_concatenated_data_filtered
from src.models.probablistic_model import estimate_probability
from src.features import Pupil_Model
from src.features.gaze_entropy import gaze_entropy
from src.features.other_gaze_features import fixation_duration

__author__ = "Regine Bueter"

# load config file
config_dict = {}
with open("src/models/config.yml", "r") as f:
    try:
        config_dict = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)


def get_pupil_response_value(path_user, use_step):
    """Get the cognitive load pupil response value for the given step.

     It loads the pupil data, calculates the difference between estimated pupil diameter
    based on a pupil model and the actual pupil diameter, and normalizes the difference
    between 0 and 1 to obtain the pupil response value for the specified steps.


    Parameters
    ----------
    path_user: str
        Processed data path to the folder of the user with the steps
    use_step: list
        Steps to use to get the pupil value

    Returns
    -------
    array-like
        The pupil value calculated for the specified steps.

    Notes
    -----
    This function uses a Pupil_Model instance specific to the user to estimate pupil
    diameter and calculate the cognitive load pupil response value based on the difference
    between estimated and actual pupil diameters.
    """
    user = Path(path_user).stem

    # Get pupil model for specific user
    pupil_model = Pupil_Model.PupilModelWithEstParams(user, load_parameters_from_file=True)

    # Load data
    pupil_diam, time_stamps, focal_brightness, ambient_light, cognit_load = load_concatenated_data_filtered(path_user, use_step)

    # get pupil diameter from model
    pupil_diam_est = pupil_model.calculate_pupil_light_response(time_stamps, focal_brightness, ambient_light)

    # Normalize difference diameter between 0 and 1
    difference_diameter = pupil_diam - pupil_diam_est
    pupil_value = (difference_diameter - np.min(difference_diameter)) / (np.max(difference_diameter) - np.min(difference_diameter))

    return pupil_value




def get_gaze_entropy(raw_data_path, processed_step_path, args):
    """Get the gaze entropy for the specified step.

    It computes the gaze entropy using processed and raw data paths for the given step
    by analyzing the gaze data within a grid of a specified size.

    Parameters
    ----------
    raw_data_path : str
        The data path to the raw data of the step to analyze.
    processed_step_path : str
        The data path to the processed data of the step to analyze.
    args : Namespace
        Additional arguments including grid_size and seconds_entropy.

    Returns
    -------
    List[float]
        A list containing gaze entropy values calculated for the specified step.

    Notes
    -----
    This function uses the gaze_entropy utility to calculate the gaze entropy based on
    processed and raw data paths. It requires 'grid_size' and 'seconds_entropy'
    parameters from the args Namespace to determine the grid size for entropy analysis
    and the duration of the segment for entropy calculation.
    """
    grid_size = args.grid_size
    gaze_ent = gaze_entropy(str(raw_data_path), str(processed_step_path), grid_size=grid_size, seconds=args.seconds_entropy)
    return gaze_ent

def get_fixation_duration(raw_data_path, processed_step_path, args):
    """Get the fixation duration for the specified step.

     It computes the fixation duration using processed and raw data paths for the given step
     by analyzing the gaze data within a grid of a specified size.

     Parameters
     ----------
     raw_data_path : str
         The data path to the raw data of the step to analyze.
     processed_step_path : str
         The data path to the processed data of the step to analyze.
     args : Namespace
         Additional arguments including grid_size and seconds_entropy.

     Returns
     -------
     List[float]
         A list containing fixation duration values calculated for the specified step.

     Notes
     -----
     This function uses the fixation_duration utility to calculate the fixation duration based on
     processed and raw data paths. It requires 'grid_size' and 'seconds_entropy'
     parameters from the args Namespace to determine the grid size for entropy analysis
     and the duration of the segment for fixation duration calculation.
     """
    grid_size = args.grid_size
    fixation_dur = fixation_duration(str(raw_data_path), str(processed_step_path), grid_size=grid_size, seconds=args.seconds_entropy)
    return fixation_dur


def get_probability_cognit_load(data_path_processed, user, steps, eye, save_path, args):
    """ Get the probability and cognitive load for the specified step.

    This function estimates the probability and cognitive load based on the provided data
    path, user, steps, eye, and save_path parameters.

    Parameters
    ----------
    data_path_processed : str
        The path to the folder with the processed data for the user.
    user : str
        The user to be analyzed.
    steps : list
        The steps to be analyzed.
    eye : str
        The eye to be analyzed.
    save_path : str
        The path to save the figures.
    args : Namespace
        Additional arguments.

    Returns
    -------
    Tuple[array-like, array-like]
        A tuple containing the probability and cognitive load arrays.

    Notes
    -----
    This function utilizes the 'estimate_probability' function to estimate the probability
    and cognitive load based on the provided parameters.
    """
    probability, cognit_load = estimate_probability(data_path_processed, user, steps, eye, save_path)
    return probability, cognit_load


def load_data(data_path_processed, raw_data_path, users, steps, eye, figure_save_path, args):
    """Load and process data for training.

    This function loads and processes data for training a machine learning model.
    It collects features (gaze entropy, pupil response value) and labels (cognitive load)
    for the specified users and steps.

    Parameters
    ----------
    data_path_processed : str
        The path to the folder with the processed data for the users.
    raw_data_path : str
        The path to the folder with the raw data.
    users : list
        List of users to be analyzed.
    steps : list
        List of steps to be analyzed.
    eye : str
        The eye to be analyzed.
    figure_save_path : str
        The path to save the figures.
    args : Namespace
        Additional arguments.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing feature matrix (X) and label vector (y) for training.

    Notes
    -----
    This function iterates over the specified users and steps to collect gaze entropy,
    pupil response value, and cognitive load labels. It then constructs feature vectors
    (X) and label vectors (y) for training a machine learning model. It also calculates
    Pearson correlation between probability and pupil value, and prints the correlation
    coefficient and its mean.
    """
    # define feature names
    X = None
    y = None
    prob_all = None
    pupil_val_all = None

    # iterate over users and steps to collect gaze entropy, pupil response value, and cognitive load labels
    for user in users:
        for step in steps:
            # get correct path for processed data and raw data for specific user and step
            processed_path_user = Path(data_path_processed) / eye / user
            raw_data_step = Path(raw_data_path) / user / step

            # get probability and cognitive load
            probability, cl = get_probability_cognit_load(processed_path_user, user, [step], eye, figure_save_path, args)
            # get gaze entropy and pupil value
            gaze_entropy = get_gaze_entropy(raw_data_step, processed_path_user / step, args)
            pupil_value = get_pupil_response_value(processed_path_user, [step])
            fixation_dur = get_fixation_duration(raw_data_step, processed_path_user / step, args)

            # make cl to binary vector, from different n-back tasks
            cl = np.asarray(cl)
            cl[cl > 0] = 1

            # make vector for sklearn training from entropy and pupil_value
            if X is None:
                X = np.array([gaze_entropy, pupil_value, fixation_dur]).T
                y = cl
            else:
                X = np.concatenate((X, np.array([gaze_entropy, pupil_value, fixation_dur]).T))
                y = np.concatenate((y, cl))
    return X, y


def visualize_data_set(X, y, feature_names):
    """Visualize the data set using pairplot.

    This function generates a pairplot to visualize the relationships between features
    (gaze entropy, pupil response value) in the data set along with the corresponding labels.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix containing the dataset's features.
    y : np.ndarray
        Label vector containing the dataset's labels.
    feature_names : List[str]
        Names of the features.


    Notes
    -----
    This function utilizes Seaborn's pairplot to create a matrix of pairwise plots for
    visualization. Each scatterplot in the pairplot matrix represents the relationship
    between different features in the dataset, differentiated by class labels.
    """
    # visualize data set
    data = pd.DataFrame(X, columns=feature_names)
    data['class'] = y
    sns.pairplot(data, hue='class')
    plt.show()


def hyperparameter_tuning(data_path_processed, raw_data_path, users, steps, eye, figure_save_path, args):
    """Perform hyperparameter tuning for a Random Forest Classifier.

    This function conducts hyperparameter tuning for a Random Forest Classifier using
    GridSearchCV to find the best set of hyperparameters based on recall scoring.

    Parameters
    ----------
    data_path_processed : str
        The path to the folder with the processed data for the users.
    raw_data_path : str
        The path to the folder with the raw data.
    users : list
        List of users to be analyzed.
    steps : list
        List of steps to be analyzed.
    eye : str
        The eye to be analyzed.
    figure_save_path : str
        The path to save the figures.
    args : Namespace
        Additional arguments.

    Returns
    -------
    dict
        The best hyperparameters found by the hyperparameter search.

    Notes
    -----
    This function loads the data using load_data, constructs a Random Forest Classifier
    model, defines a grid of hyperparameters, and performs hyperparameter tuning using
    GridSearchCV to find the best combination of hyperparameters based on recall scoring.
    The best hyperparameters are printed and returned as a dictionary.
    """
    # load hyperparameter search data
    X, y = load_data(data_path_processed, raw_data_path, users, steps, eye, figure_save_path, args)
    inner_cv = sklearn.model_selection.KFold(n_splits=len(users), shuffle=False)

    # define hyperparameter grid
    param_grid = {'n_estimators': np.arange(20, 500, 10),
                  'max_depth': [None] + np.arange(1, 50).tolist(),
                  'min_samples_split': np.arange(1, 50)}

    # Hyperparameter search
    model = RandomForestClassifier()
    hyperparameter_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv, n_jobs=5,
                                         scoring='recall', verbose=1)

    hyperparameter_search.fit(X, y)
    print("Best Hyperparameters: ", hyperparameter_search.best_params_)
    return hyperparameter_search.best_params_

def train_cross_fold(data_path_processed, raw_data_path, users, steps, eye, figure_save_path, model, model_save_path, args):
    """Train a model using leave-one-user-out cross-validation and save model trained additionally on all data.

    This function trains a machine learning model using leave-one-user-out cross-validation
    and saves the trained model to the specified path.

    Parameters
    ----------
    data_path_processed : str
        The path to the folder with the processed data for the users.
    raw_data_path : str
        The path to the folder with the raw data.
    users : list
        List of users to be analyzed.
    steps : list
        List of steps to be analyzed.
    eye : str
        The eye to be analyzed.
    figure_save_path : str
        The path to save the figures.
    model : object
        The machine learning model to be trained.
    model_save_path : str
        The path to save the trained model.
    args : Namespace
        Additional arguments.

    Returns
    -------
    object
        The trained machine learning model.

    Notes
    -----
    This function uses leave-one-user-out cross-validation to train the provided model
    and then saves the trained model using pickle to the specified path. It also prints
    the True Positive Rate (TPR) and Negative Predictive Value (NPV) scores averaged
    across folds during cross-validation.
    """
    # define leave one user out cross validation
    outer_cv = sklearn.model_selection.KFold(n_splits=len(users), shuffle=False)

    # load data
    X, y = load_data(data_path_processed, raw_data_path, users, steps, eye, figure_save_path, args)


    scores = sklearn.model_selection.cross_val_score(model, X, y, scoring='recall', cv=outer_cv, n_jobs=5)
    print('TPR: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    npv = make_scorer(negative_predictive_value, greater_is_better=True)
    scores = sklearn.model_selection.cross_val_score(model, X, y, scoring=npv, cv=outer_cv, n_jobs=-1)
    print('NPV: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    # train model with all data from protocol 1 without Peg-Transfer
    model.fit(X, y)
    # save model
    with open(Path(model_save_path) / f'cross_fold_model_{args.seconds_entropy}.pkl', 'wb') as file:
        pickle.dump(model, file)
    return model

def train_model(data_path_processed, raw_data_path, users, steps, eye, figure_save_path, model, model_save_path, args):
    """Train a interpretable model.

    This function loads the data, trains the provided model using the loaded data,
    and saves the trained model to the specified path.

    Parameters
    ----------
    data_path_processed : str
        The path to the folder with the processed data for the users.
    raw_data_path : str
        The path to the folder with the raw data.
    users : list
        List of users to be analyzed.
    steps : list
        List of steps to be analyzed.
    eye : str
        The eye to be analyzed.
    figure_save_path : str
        The path to save the figures.
    model : object
        The machine learning model to be trained.
    model_save_path : str
        The path to save the trained model.
    args : Namespace
        Additional arguments.

    Returns
    -------
    object
        The trained machine learning model.

    Notes
    -----
    This function loads the data using load_data, trains the provided model with
    the loaded data, and saves the trained model using pickle to the specified path.
    The trained model object is returned.
    """
    # load data
    X, y = load_data(data_path_processed, raw_data_path, users, steps, eye, figure_save_path, args)

    # train model with all data from protocol 1 without Peg-Transfer
    model.fit(X, y)
    # save model
    with open(Path(model_save_path) / f'cross_fold_model_{args.seconds_entropy}.pkl', 'wb') as file:
        pickle.dump(model, file)
    return model

def predict_model(data_path_processed, raw_data_path, users, steps, eye, figure_save_path, model, protocol, args):
    """Predict using the trained model and visualize SHAP values for different cases.

    This function loads data, predicts using the provided model, calculates recall score,
    negative predictive value (if 'no' in protocol), and visualizes SHAP values for false
    positive, true negative, and false negative cases.

    Parameters
    ----------
    data_path_processed : str
        The path to the folder with the processed data for the users.
    raw_data_path : str
        The path to the folder with the raw data.
    users : list
        List of users to be analyzed.
    steps : list
        List of steps to be analyzed.
    eye : str
        The eye to be analyzed.
    figure_save_path : str
        The path to save the figures.
    model : object
        The trained machine learning model.
    protocol : str
        The protocol type.
    args : Namespace
        Additional arguments.

    Notes
    -----
    This function loads the data using load_data, predicts using the provided model,
    calculates recall score, negative predictive value (if 'no' in protocol), and
    visualizes SHAP values for false positive, true negative, and false negative cases.
    The generated SHAP value plots are saved as images in the specified figure save path.
    """
    # load data and predict model
    X, y = load_data(data_path_processed, raw_data_path, users, steps, eye, figure_save_path, args)
    y_pred = model.predict(X)

    # get tpr rate
    print(f"tpr : {recall_score(y, y_pred)}")
    if 'no' in protocol:
        tp, fn, fp, tn = confusion_matrix(y, y_pred).ravel()
        print(f'negative predictive value: {tn / (tn + fn)}')

    # setup for shap values
    explainer = shap.TreeExplainer(model)
    features_names = ['gaze_entropy', 'pupil_value', 'fixation_duration']

    # shap values for false positive cases
    X_fp = X[list(set(np.where(y_pred == 1)[0].tolist()) & set(np.where(y == 0)[0].tolist()))]
    shaply_values_fp = explainer.shap_values(X_fp)
    shap.summary_plot(shaply_values_fp[1], X_fp, feature_names=features_names, show=False)
    plt.savefig(f'{figure_save_path}/shaply_values_fp_{protocol}_{args.seconds_entropy}.png', dpi=700)

    # shap values for true negative cases
    # X_tn = X[list(set(np.where(y_pred == 0)[0].tolist()) & set(np.where(y == 0)[0].tolist()))]
    # shaply_values_tn = explainer.shap_values(X_tn)
    # shap.summary_plot(shaply_values_tn[0], X_tn, feature_names=features_names, show=False)
    # plt.savefig(f'{figure_save_path}/shaply_values_tn_{protocol}_{args.seconds_entropy}.png', dpi=700)

    # # shap values for false negative cases
    # X_fn = X[list(set(np.where(y_pred == 0)[0].tolist()) & set(np.where(y == 1)[0].tolist()))]
    # shaply_values_fn = explainer.shap_values(X_fn)
    # shap.summary_plot(shaply_values_fn[0], X_fn, feature_names=features_names, show=False)
    # plt.savefig(f'{figure_save_path}/shaply_values_fn_{protocol}_{args.seconds_entropy}.png', dpi=700)


def negative_predictive_value(y_true, y_pred):
    """Calculate the Negative Predictive Value (NPV).

    NPV is a measure used to evaluate the performance of a binary classification
    model. It represents the proportion of true negative predictions among all
    instances that are predicted as negative.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels.

    Returns
    -------
    float
        The calculated Negative Predictive Value (NPV).

    Notes
    -----
    This function computes the NPV using the confusion matrix derived from
    true labels (y_true) and predicted labels (y_pred). It calculates the ratio
    of true negatives to the sum of true negatives and false negatives.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn)


def main(args):
    # train parameters
    raw_data_path = args.folder_path_raw
    eye = args.eye
    figure_save_path = args.figure_save_path
    model_save_path = args.model_save_path

    # load or find best hyper parameters
    if args.hyperparameter_tuning:
        print("[INFO] Hyperparameter tuning...")
        # load hyperparameter search data from config file
        data_path_hp = config_dict['data_path_hyperparameter_tuning']
        users_hp = config_dict['users_hyperparameter_tuning']
        steps = config_dict['steps_hyperparameter_tuning']

        # perform hyperparameter tuning
        best_params = hyperparameter_tuning(data_path_hp, raw_data_path, users_hp, steps, eye, figure_save_path, args)

        # save best params to file system
        with open(Path(model_save_path) / f'best_params_interpretable_{args.seconds_entropy}.pkl', 'wb') as file:
            pickle.dump(best_params, file)
    else:
        print("[INFO] Load best hyper parameters...")
        # load best hyper parameters from file system
        with open(Path(model_save_path) / f'best_params_interpretable_{args.seconds_entropy}.pkl', 'rb') as file:
            best_params = pickle.load(file)



    # train model or load model
    if args.train or args.train_cross_fold:
        # define model with best parameters
        model = RandomForestClassifier(**best_params)

        # train model
        train_data_path = args.train_data_path
        train_users = args.train_users
        train_steps = args.train_steps

        if args.train_cross_fold:
            print("[INFO] Train model with leave one user out cross fold validation...")
            # train model with leave one user out cross fold validation
            model = train_cross_fold(train_data_path, raw_data_path, train_users, train_steps, eye, figure_save_path, model, model_save_path, args)
        else:
            print("[INFO] Train model...")
            # train model
            model = train_model(train_data_path, raw_data_path, train_users, train_steps, eye, figure_save_path, model, model_save_path, args)
    else:
        print("[INFO] Load model...")
        # load model
        with open(Path(model_save_path) / f'cross_fold_model_{args.seconds_entropy}.pkl', 'rb') as file:
            model = pickle.load(file)

    # predict model
    print("[INFO] Predict model...")
    test_data_path = args.test_data_path
    test_users = args.test_users
    test_steps = args.test_steps
    trial_save_name = args.trial_save_name
    predict_model(test_data_path, raw_data_path, test_users, test_steps, eye, figure_save_path, model, trial_save_name, args)


def process_arguments(args):
    """Process the arguments for data and model training.

    This function checks if a trained model exists when it shouldn't be trained.
    It loads trial files for both training and testing, extracts users and steps from them,
    and checks if these users and steps exist in the file system.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments.

    Returns
    -------
    Namespace
        Processed arguments containing extracted user and step information.

    Raises
    ------
    FileNotFoundError
        If the trained model or necessary files (train trial, test trial) are not found.

    Notes
    -----
    This function processes the arguments related to data paths, eye type, training, and test data.
    It checks for the existence of required files, extracts users and steps from the trial files,
    and populates 'train_users', 'train_steps', 'test_users', and 'test_steps' in the args namespace.
    """

    # check if a trained model exists, if model should not be trained
    if not args.train or args.train_cross_fold:
        if not Path(Path(args.model_save_path) / f'cross_fold_model_{args.seconds_entropy}.pkl').exists():
            raise FileNotFoundError("No trained model found, please train model first")

    # load train trial file if training is performed
    if args.train or args.train_cross_fold:
        if not Path(args.train_trial).exists():
            raise FileNotFoundError
        with open(args.train_trial) as file:
            train_data = [line.rstrip() for line in file if len(line.rstrip()) != 0]

        # get train users and steps and check if they exist in file system
        args.train_users = []
        args.train_steps = []

        # get all users in train folder
        all_users_train_path = Path(args.train_data_path) / args.eye
        all_users_train = sorted(os.listdir(all_users_train_path))
        for user in all_users_train:
            if user == ".DS_Store" or '._' in user:
                continue
            if user in train_data:
                args.train_users.append(user)

            # get all steps in train folder and check if they exist in file system, if so add to train steps
            all_steps = os.listdir(all_users_train_path / user)
            for step in all_steps:
                if step in train_data and step not in args.train_steps:
                    args.train_steps.append(step)

    # load test trial
    if not Path(args.trial).exists():
        raise FileNotFoundError
    with open(args.trial) as file:
        test_data = [line.rstrip() for line in file if len(line.rstrip()) != 0]

    # get all test users and steps
    args.test_users = []
    args.test_steps = []

    # get all users in train folder
    all_users_test_path = Path(args.test_data_path) / args.eye
    all_users_test = sorted(os.listdir(all_users_test_path))
    for user in all_users_test:
        if user == ".DS_Store" or '._' in user:
            continue
        if user in test_data:
            args.test_users.append(user)

        # get all steps in train folder and check if they exist in file system, if so add to train steps
        all_steps = os.listdir(all_users_test_path / user)
        for step in all_steps:
            if step in test_data and step not in args.test_steps:
                args.test_steps.append(step)
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for Probabilistic Model")
    # paths to data, interim data and results
    parser.add_argument("--train_data_path", type=str, default="data/processed/ipcai_2024/protocol_1",
                        help="Path to the folder with the processed data for the users.")
    parser.add_argument("--test_data_path", type=str, default="data/processed/ipcai_2024/protocol_1",
                       help="Path to the folder with the processed data for the users to test.")
    parser.add_argument("--folder_path_raw", required=False, type=str, default="data/raw/ipcai_2024/pupilcapture/protocol_1",
                        help="Path to the folder containing the raw data")
    parser.add_argument("--model_save_path", type=str, default='models',
                        help="Path to save the model")
    parser.add_argument("--figure_save_path", type=str, default='reports',
                        help="Path to save the figures")
    parser.add_argument("--train_trial", type=str, default='from_file',
                        help='The usable trials for the training can be loaded from a file. The txt file should '
                             'contain the user in each line and the steps to be used for training in each line.')
    parser.add_argument("--trial", type=str, default='from_file',
                        help='The usable trials for the prediction can be loaded from a file. The txt file should '
                             'contain the user in each line and the steps to be used for training in each line.')
    parser.add_argument("--trial_save_name", type=str, default="",
                        help="Name of the trial to save the results under the correct name.")


    parser.add_argument("--hyperparameter_tuning", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="If True, hyperparameter tuning is performed, otherwise the best model is loaded")
    parser.add_argument("--train", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="If True, training is performed, otherwise only prediction")
    parser.add_argument("--train_cross_fold", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="If True, training is performed with a leave one user out cross fold validation, "
                             "otherwise only training on the data is performed")






    parser.add_argument("--individual_steps", type=bool, default=False, action=argparse.BooleanOptionalAction)
    # other parameters
    parser.add_argument("--seconds_entropy", type=int, default=10)
    parser.add_argument("--eye", type=str, default="right")

    parser.add_argument("--grid_size", required=False, type=int, default=15)
    args = parser.parse_args()


    args = process_arguments(args)

    main(args)
