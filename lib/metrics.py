import numpy as np


def RSE(y_pred, y_label):
    """
    Relative Squared Error (RSE)
    RSE = ∑_(i=1)^n (yi-yi_pred)^2 / ∑_(i=1)^n (yi-y_mean)^2
    """
    RSE = np.sqrt(np.sum((y_label - y_pred) ** 2)) / np.sqrt(np.sum((y_label - y_label.mean()) ** 2))

    return RSE


def CORR(y_pred, y_label):
    """
    Spearman Correlation coefficient
    """
    u = ((y_label - y_label.mean(0)) * (y_pred - y_pred.mean(0))).sum(0)
    d = np.sqrt(((y_label - y_label.mean(0)) ** 2).sum(0) * ((y_pred - y_pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(y_pred, y_label):
    return np.mean(np.abs(y_pred - y_label))


def MSE(y_pred, y_label):
    return np.mean((y_pred - y_label) ** 2)


def RMSE(y_pred, y_label):
    return np.sqrt(MSE(y_pred, y_label))


def MAPE(y_pred, y_label):
    """
    MAPE= 1/n ∑_{i=1}^n |yi_label -yi_pred|/yi_label
    """
    return np.mean(np.abs((y_pred - y_label) / y_label))


def MSPE(y_pred, y_label):
    return np.mean(np.square((y_pred - y_label) / y_label))


def metric(y_pred, y_label):
    mae = MAE(y_pred, y_label)
    mse = MSE(y_pred, y_label)
    rmse = RMSE(y_pred, y_label)
    mape = MAPE(y_pred, y_label)
    mspe = MSPE(y_pred, y_label)
    rse = RSE(y_pred, y_label)
    corr = CORR(y_pred, y_label)

    return mae, mse, rmse, mape, mspe, rse, corr
