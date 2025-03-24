import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from safepython import HyMod

param_names = ["SMAX", "BETA", "ALPHA", "RS", "RF"]
lower       = np.array([0     , 0     , 0      , 0   , 0.5])
upper       = np.array([400   , 2     , 1      , 0.5 , 1])
ranges      = np.stack([lower , upper], axis = 1)
n_params    = len(param_names)


def load_catchment_data(file="02472000.txt", startDate="2010", endDate="2015"):
    """
    Load catchment data from a file and subset it to a given date range.
    The file must have the columns "YR", "MNTH", "DY", "PRCP", "TAIR", "PET" and "OBS_RUN".
    """
    data = pd.read_csv(file, sep=r"\s+")
    # combine columns to create a datetime column
    data['DATE'] = pd.to_datetime(data[['YR', 'MNTH', 'DY']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d')
    # set the new column as the index
    data = data.set_index('DATE')
    # keep only relevant variables
    data = data[["PRCP", "TAIR", "PET", "SWE", "OBS_RUN"]]
    # rename the columns
    data.columns = ["P", "T", "PET", "SWE", "Q_obs"]
    # subset range
    data = data.loc[(data.index >= startDate) & (data.index <= endDate)]
    data.file = file
    return data


def plot_hydrograph(data):
    """
    Plot the hydrograph of a catchment with observed runoff, precipitation and snow volume.
    """
    h1 = plt.plot(data.index, data["Q_obs"], color="black")
    plt.xlabel("Date")
    plt.ylabel("Runoff [mm/day]")
    plt.sca(plt.twinx())
    plt.gca().invert_yaxis()
    h2 = plt.fill_between(data.index, 0, data["SWE"]/data["SWE"].max()*data["P"].max(), color="gray", alpha=0.5)
    h3 = plt.bar(data.index, data["P"], width=4)
    plt.ylim([2*plt.ylim()[0], 0])
    plt.ylabel("Precipitation [mm/day]")
    handles = [*h1, h2, *h3]
    labels  = ["Observed Runoff", "Snow Volume", "Precipitation"]
    plt.legend(handles, labels)


def plot_indices(x, **kwargs):
    kwargs.setdefault("widths", 0.4)
    kwargs.setdefault("patch_artist", True)
    kwargs.setdefault("showfliers", False)
    kwargs.setdefault("positions", range(x.shape[1]))
    boxstyle = kwargs.pop("boxstyle", {})
    plot = plt.boxplot(x, **kwargs)
    for i, box, median in zip(range(len(x)), plot["boxes"], plot["medians"]):
        box.set(facecolor=f"C{i}", **boxstyle)
        median.set_color("black")
        median.set_linewidth(2)


def run_hymod(data, params):
    """
    Run the HBV model on a given dataset and parameter set.

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame containing the columns "P", "T", "PET" and "Q_obs". The columns "P" and "T" are precipitation and temperature, respectively. "PET" is the potential evapotranspiration and "Q_obs" is the observed discharge.
    params: np.array of shape (13,)
        Array containing the HBV parameters in the following order: Ts, CFMAX, CFR, CWH, BETA, LP, FC, PERC, K0, K1, K2, UZL, MAXBAS
    """
    # check that there is the correct number of parameters
    assert params.shape == (5,)
    # convert PRCP, TAIR and PET to numpy arrays if necessary
    P, PET = data[["P", "PET"]].T.to_numpy()
    # run HBV snow and rainfall-runoff routine
    Q, _, _ = HyMod.hymod_sim(params, P, PET)
    return Q


def exclude_spinup(of, spinup=0.1):
    """
    Decorator to exclude the spinup period from the objective function calculation.

    Parameters
    ----------
    of : function
        Objective function to be wrapped. Must have the signature of (Q_obs, Q_sim)
    spinup: float or int
        Fraction of the spinup period to be excluded from the objective function calculation. If an integer is given, the first n values are excluded. Default is 0.1.
    """
    from functools import wraps
    @wraps(of)
    def _exclude_spinup_wrapper(Q_obs, Q_sim):
        assert Q_obs.shape == Q_sim.shape
        index = int(spinup) if spinup < 1 else int(spinup * len(Q_obs))
        return of(Q_obs[index:], Q_sim[index:])
    return _exclude_spinup_wrapper


@exclude_spinup
def absbias(Q_obs, Q_sim):
    """
    Calculate the absolute bias between observed and simulated discharge.
    """
    return np.abs(np.mean(Q_sim - Q_obs))


@exclude_spinup
def rmse(Q_obs, Q_sim):
    """
    Calculate the root mean squared error between observed and simulated discharge.
    """
    return np.mean((Q_sim - Q_obs)**2)**0.5


@exclude_spinup
def flow_mean(Q_obs, Q_sim):
    """
    Calculate the mean flow of the observed discharge.
    """
    return Q_sim.mean()


@exclude_spinup
def flow_std(Q_ob, Q_sim):
    """
    Calculate the standard deviation of the observed discharge.
    """
    return Q_sim.std()