import keras.backend as K
from keras.optimizers import Adam
from keras.initializers import he_uniform, he_normal

from hyperopt import hp
from hyperopt.pyll import scope

from utilities_dnn import u_net

def dice_loss(y_true, y_pred, smooth=1.e-6):
    #inputs = K.flatten(y_pred)
    #targets = K.flatten(y_true)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    #intersection = K.sum(K.dot(targets, inputs))
    #dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    dice = (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    return 1 - dice


def make_model_config(grid_phi, grid_r, grid_z, dim_input,
                      depth, batch_normalization, pooling, filters, dropout,
                      loss, learning_rate, epochs):
    return {"constructor": u_net,
            "model_args": ((grid_phi, grid_r, grid_z, dim_input),),
            "model_kwargs": {"depth": depth,
                             "batchnorm": batch_normalization,
                             "pool_type": pooling,
                             "start_channels": filters,
                             "dropout": dropout},
            "compile": {"loss": "mse",
                        "optimizer": {"callable__": Adam, "kwargs": {"lr": learning_rate}},
                        "metrics": [loss, dice_loss]},
            "fit": {"workers": 20,
                    "use_multiprocessing": True,
                    "epochs": epochs}}


def make_opt_space():
    return {"compile": 
            {"optimizer": {"kwargs": {"lr": hp.uniform("m_learning_rate", 0.0005, 0.01)}},
             "loss": hp.choice("m_loss", ["mse", "mean_squared_logarithmic_error"])},
             "model_kwargs": {"batchnorm": hp.choice("m_bathnorm", [0, 1]),
                              "dropout": hp.uniform("m_dropout", 0., 0.5),
                              "inc_rate": scope.int(hp.quniform("m_inc_rate", 1, 4, 1)),
                              "start_channels": scope.int(hp.quniform("m_start_channels", 2, 8, 1)),
                              "depth": scope.int(hp.quniform("m_depth", 2, 4, 1))},
             "fit": {"epochs": scope.int(hp.quniform("m_epochs", 2, 4, 1))}}
