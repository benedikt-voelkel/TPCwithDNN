import sys
from os.path import exists, join, split
from os import makedirs, getpid
from copy import deepcopy

from datetime import time

from yaml.representer import RepresenterError
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping

from machine_learning_hep.io import parse_yaml, dump_yaml_from_dict, dict_yamlable
from machine_learning_hep.logger import get_logger


# Common file names
NAME_MODEL_ARCH = "model.json"
NAME_MODEL_WEIGHTS = "weights.h5"
NAME_MODEL_CONFIG = "config.yaml"


def digest_model_config(model_config):
    """call callables with given args and kwargs and add inplace 

    Args:
        model_config: dict
            model configuration to digest
    """
    for k in list(model_config.keys()):
        curr = model_config[k]
        if isinstance(curr, dict):
            callable__ = curr.get("callable__", None)
            if callable__:
                model_config[k] = callable__(*curr.get("args", []), **curr.get("kwargs", {}))
                continue
            digest_model_config(model_config[k])


def compile_model(model, compile_config):
    """comile model

    Args:
        model: Keras TF model
            model to be compiled
        model_config: dict
            dictionary with key "compile" hodling keyword args
    """
    model.compile(**compile_config)
    model.summary()


def construct_model(constructor, *args, **kwargs):
    """construct model from given configuration

    Args:
        constructor: callable
            returning a model digesting given configuration
        *args: arguments
            arguments to be forwarded to constructor
        **kwargs: keyword arguments
            keyword arguments to be forwarded to constructor

    Returns: model
    """

    model_config = None
    if not args and not kwargs:
        # Assume it's a model_config dictionary
        get_logger().info("Construct model from config dictionary")
        model_config = constructor

    model_args = args
    model_kwargs = kwargs
    if model_config:
        model_args = model_config.get("model_args", model_args)
        model_kwargs = model_config.get("model_kwargs", model_kwargs)

    model = model_config["constructor"](*model_args, **model_kwargs)
    compile_model(model, model_config.get("compile", {}))
    return model


def fit_model(model, x=None, y=None, weights=None, gen=None, val_data=None, test_data=None,
              **fit_kwargs):
    """Fit the model

    Fit model from data directly or from data generators

    Args:
        model: Keras TF model
            model to be fit
        x: n_samples x n_features numpy array like (optional)
            training features
        y: n_samples x n_targets numpy array like (optional)
            training targets
        weights: n_samples x 1 numpy array like (optional)
            training sample weights
        gen: n_samples x (n_features, n_targets, [weights]) numpy array like generator (optional)
            training data provided by generator (optionally including sample weights)
        val_data: tuple or generator of (n_samples_val x n_features, n_samples_val x n_targets)
                  numpy array like (optional)
            validation data to be used
        test_data: tuple or generator of (n_samples_val x n_features, n_samples_val x n_targets)
                   numpy array like (optional)
            test data to be used
        fit_kwargs: dict
            additional arguments passed to the model's fit  method

    Notes:
        EITHER x and y (and optionally weights) are specified OR gen is given.
    """
    if test_data:
        print("Assume model has been trained already ==> predict on test data")
        return

    # First do generic checks if there is data to fit to
    if not x and not y and not gen:
        get_logger().fatal("Need either arguments \"x\" and \"y\" or \"gen\"")
    if y and not x:
        get_logger().fatal("\"y\" is given but \"x\" is missing")

    # Decide on the fit method, if gen is set superseeds x and y
    fit_method = None

    if gen:
        fit_method = "fit_generator"
        fit_kwargs["generator"] = gen
    else:
        fit_method = "fit"
        fit_kwargs["x"] = x
        fit_kwargs["y"] = y
        fit_kwargs["sample_weight"] = weights
    if val_data:
        fit_kwargs["validation_data"] = val_data

    if not hasattr(model, fit_method):
        get_logger().fatal("Given model has no method %s", fit_method)

    return getattr(model, fit_method)(**fit_kwargs)


def make_out_dir(out_dir, overwrite=False, suffix=0):
    """make an output directory

    Try to make specified output directory. If it exists, add suffix "_i"
    and increment "i" until a name is found which doesn't exist. The first
    directory possible is created.

    Args:
        out_dir: str
            desired output directory path
        overwrite: bool
            whether or not just overwrite/write into an existing directory
        suffix: int (optional)
            suffix to start incrementing from in case desired out_dir exists
    Returns:
        str: output directory path

    """
    if not exists(out_dir):
        makedirs(out_dir)
        return out_dir
    if overwrite:
        return out_dir
    dir_name, base_name = split(out_dir)
    base_name = f"{base_name}_{suffix}"
    suffix += 1
    return make_out_dir(join(dir_name, base_name), suffix)


def save_model(model, model_config, out_dir, overwrite=False):
    """save a model

    Save a model and the configuration used to create it. This creates a JSON file
    containing the architecture, an HDF5 file with weights and a YAML file with
    the model configuration.

    Args:
        model: Keras/TF model
            model to be saved
        model_config: dict
            model configuration the model was created from within this package
    """
    out_dir = make_out_dir(out_dir, overwrite=True)
    get_logger().info("Save model config and weights at %s", out_dir)
    model_json = model.to_json()
    save_path = join(out_dir, NAME_MODEL_ARCH)
    with open(save_path, "w") as json_file:
        json_file.write(model_json)
    save_path = join(out_dir, NAME_MODEL_WEIGHTS)
    model.save_weights(save_path)
    try:
        save_path = join(out_dir, NAME_MODEL_CONFIG)
        dump_yaml_from_dict(dict_yamlable(model_config), save_path)
    except RepresenterError:
        print("Cannot save model configuration as YAML")


def load_model(in_dir):
    """load saved model from directory

    Load a model that was saved before with save_model

    Args:
        in_dir: str
            directory path where model was saved
    Returns:
        Keras/TF model,
        dict: model configuration the model was created from within this package
    """
    if not exists(in_dir):
        get_logger().fatal("Directory %s does not exist. Cannot load model", in_dir)
    json_path = join(in_dir, NAME_MODEL_ARCH)
    weights_path = join(in_dir, NAME_MODEL_WEIGHTS)
    config_path = join(in_dir, NAME_MODEL_CONFIG)
    if not exists(json_path) or not exists(weights_path):
        get_logger().fatal("Make sure there is there are all files to load a model from: %s",
                           str((json_path, weights_path)))

    model = None
    with open(json_path, "r") as json_file:
        model_arch = json_file.read()
        model = model_from_json(model_arch)
    model.load_weights(weights_path)
    model_config = None
    if exists(config_path):
        model_config = parse_yaml(config_path)

    return model, model_config


def bayesian_trial(model, x_train, y_train, fit_params, config):
    """One trial

    This does one fit attempt (no CV at the moment) with the given parameters

    """

    # Check if scoring is in metrics list and add if not present
    """
    if config.scoring not in model_config["compile"]["metrics"]:
        is_callable = False
        for m in model_config["compile"]["metrics"]:
            if callable(m) and m.__name__ == config.scoring:
                # There is sometnig which is callable and has the name of the desired scoring, good
                is_callable = True
                break
        if not is_callable:
            # If it was not there as a callable, add the name from config and hope keras can handle
            # that...
            model_config["compile"]["metrics"].append(config.scoring)
    """

    time_stamp = time()

    mode = "min" if config.lower_is_better else "max"
    checkpoint_weights_path = f"/tmp/BayesianOpt_fit_{time}_{{epoch:02d}}-{{val_{config.scoring}:.4f}}"
    checkpoint = ModelCheckpoint(filepath=checkpoint_weights_path,
                                 save_weights_only=True,
                                 monitor=f"val_{config.scoring}",
                                 mode=mode,
                                 save_best_only=True)

    early_stopping = EarlyStopping(monitor=f"val_{config.scoring}",
                                   patience=3,
                                   mode=mode)
    val_gen = config.get_attachment("val_gen")
    history = fit_model(model, gen=x_train, val_data=val_gen, **fit_params, callbacks=[checkpoint, early_stopping])


    # Evaluate the model
    predict_kwargs = {k: v for k, v in fit_params.items() if k in ["workers", "use_multiprocessing"]}
    score_train = None
    score_test = None
    #test_data = self.test_data if self.test_data else self.val_gen
    scores_train = model.evaluate(x_train, verbose=1, **predict_kwargs)
    scores_test = model.evaluate(val_gen, verbose=1, **predict_kwargs)
    for ln, lv_train, lv_test in zip(model.metrics_names, scores_train, scores_test):
        if ln == config.scoring:
            score_train = lv_train
            score_test = lv_test
            break

    #model.load_weights(checkpoint_weights_path)
    return config.make_results([score_train], [score_test])
