# -*- coding: utf-8 -*-
"""
Created on Tue May 11 10:23:05 2021

Fonctions utiles pour utiliser python en régu

@author: Alban Van Laethem
"""

import math  # Library to be able to do some mathematical operations
import control as ct
from control import (
    matlab as ml,
)  # Python Control Systems Toolbox (compatibility with MATLAB)

# from control.matlab import *
# from control.freqplot import default_frequency_range
from control.nichols import nichols_grid
import numpy as np  # Library to manipulate array and matrix
import matplotlib.pyplot as plt  # Library to create figures and plots

# import scipy as sp
from matplotlib.offsetbox import AnchoredText  # To print text inside a plot

# Package to make interactive plots
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import control_plotly as cp  # Package to plot directly control figures using Plotly

# List of the Plotly (D3) standard colors
color_list = px.colors.qualitative.Plotly

# %% Object to store all the important informations provided by the step
class Info:

    """
    Object to store all the interesting informations provided by the step
    response.

    Attributes
    ----------

    RiseTime: float
        Time it takes for the response to rise from 10% to 90% of the
        steady-state response.

    SettlingTime: float
        Time it takes for the error e(t) = \|y(t) – yfinal\| between the
        response y(t) and the steady-state response yfinal to fall below 5% of
        the peak value of e(t).

    SettlingMin: float
        Minimum value of y(t) once the response has risen.

    SettlingMax: float
        Maximum value of y(t) once the response has risen.

    Overshoot: float
        Percentage overshoot, relative to yfinal.

    Undershoot: float
        Percentage undershoot.

    Peak: float
        Peak absolute value of y(t).

    PeakTime: float
        Time at which the peak value occurs.

    DCGain: float
        Low-frequency (DC) gain of LTI system.
    """

    DCGain = None
    RiseTime = None
    SettlingTime = None
    SettlingMin = None
    SettlingMax = None
    Overshoot = None
    Undershoot = None
    Peak = None
    PeakTime = None


# %% printInfo
def printInfo(info):
    """
    Print in alphabetical order the informations stored in the given Info
    object.

    Parameters
    ----------
    info: Info
        Object in which all the informations of the step response are stored.

    Returns
    -------
    None
    """

    # Transform into a dict to be able to iterate
    temp = vars(info)
    for item in sorted(temp):
        print(item, ":", temp[item])

    # print("RiseTime:", info.RiseTime)
    # print("SettlingTime:", info.SettlingTime)
    # print("SettlingMin:", info.SettlingMin)
    # print("SettlingMax:", info.SettlingMax)
    # print("Overshoot:", info.Overshoot)
    # print("Undershoot:", info.Undershoot)
    # print("Peak:", info.Peak)
    # print("PeakTime:", info.PeakTime)
    # print("DCGain:", info.DCGain)


# %% pzmap
# Static pzmap plot reviewed by Alban Van Laethem
def pzmap_(
    sys, plot=True, title="Pole Zero Map", NameOfFigure="", sysName="", color="b"
):
    """
    Plot a pole/zero map for a linear system.

    Parameters
    ----------
    sys: LTI (StateSpace or TransferFunction)
        Linear system for which poles and zeros are computed.
    plot: bool, optional
        If ``True`` a graph is generated with Matplotlib,
        otherwise the poles and zeros are only computed and returned.
    NameOfFigure: String, optional
        Name of the figure in which plotting the step response.
    sysName: String, optional
        Name of the system to plot.

    Returns
    -------
    poles: array
        The systems poles
    zeros: array
        The system's zeros.
    """

    poles = sys.pole()
    zeros = sys.zero()

    if plot:
        if NameOfFigure == "":
            plt.figure()
        else:
            plt.figure(NameOfFigure)

        # Plot the locations of the poles and zeros
        if len(poles) > 0:
            plt.scatter(
                poles.real,
                poles.imag,
                s=50,
                marker="x",
                label=sysName,
                facecolors=color,
            )
        if len(zeros) > 0:
            plt.scatter(
                zeros.real,
                zeros.imag,
                s=50,
                marker="o",
                label=sysName,
                facecolors=color,
                edgecolors="k",
            )

        plt.title(title)
        plt.ylabel("Imaginary Axis (1/seconds)")
        plt.xlabel("Real Axis (1/seconds)")
        if sysName != "":
            plt.legend()

    # Return locations of poles and zeros as a tuple
    return poles, zeros


# Interactive pzmap plot
def pzmap(sys_list, plot=True, title="Pole Zero Map", fig=None):
    """
    Plot a pole/zero map for a linear system.

    Parameters
    ----------
    sys_list: list of LTI, or LTI
        List of linear input/output systems.
    plot: bool, optional
        If ``True`` a graph is generated with Matplotlib,
        otherwise the poles and zeros are only computed and returned.
    fig: plotly.graph_objs.Figure
        The figure where to plots the poles and zeros

    Returns
    -------
    poles: array
        The systems poles
    zeros: array
        The system's zeros.
    """

    # Convert the given system to a list if there is only one
    if type(sys_list) is not list:
        sys_list = [sys_list]

    # Create the figure to plot the traces
    if fig is None:
        fig = go.Figure()
        # Define the figure layout
        fig.layout = go.Layout(
            title=title,
            xaxis=dict(title="Real Axis"),
            yaxis=dict(title="Imaginary Axis"),
            # template="plotly_white",
            legend=dict(groupclick="toggleitem"),
            showlegend=True,
        )
    else:  # Remove the previous plots if the figure already exist
        fig.data = []

    for sys_index, sys in enumerate(sys_list):
        poles = sys.pole()
        zeros = sys.zero()

        # Plot the locations of the poles and zeros
        if len(poles) > 0:
            trace_poles = go.Scatter(
                x=np.real(poles),
                y=np.imag(poles),
                mode="markers",
                marker=dict(
                    color=color_list[sys_index],
                    size=10,
                    symbol="x-thin",
                    line_width=2,
                    line_color=color_list[sys_index],
                ),
                name="Poles",
                legendgroup="sys{}".format(sys_index+1),
                legendgrouptitle_text="sys{}".format(sys_index+1),
            )
            fig.add_trace(trace_poles)

        if len(zeros) > 0:
            trace_zeros = go.Scatter(
                x=np.real(zeros),
                y=np.imag(zeros),
                mode="markers",
                marker=dict(
                    color="rgba(255, 255, 255, 0)",
                    size=10,
                    symbol="circle",
                    line_width=2,
                    line_color=color_list[sys_index],
                ),
                name="Zeros",
                legendgroup="sys{}".format(sys_index+1),
            )
            fig.add_trace(trace_zeros)

    # Show the figure
    if plot:
        fig.show()

    # Return the figure and the locations of poles and zeros as a tuple
    return fig, poles, zeros


# Interactive pzmap plot with widgets compatibility (Works only with Jupyter Notebooks !)
def pzmap2(sys_list, plot=False, title="Pole Zero Map", fig=None):
    """
    Plot a pole/zero map for a linear system.

    Parameters
    ----------
    sys_list: list of LTI, or LTI
        List of linear input/output systems.
    plot: bool, optional
        Show the figure if True (Default = False)
    fig: go.FigureWidget
        The figure where to plots the poles and zeros

    Returns
    -------
    fig: plotly.graph_objs.FigureWidget
        The figure where the poles and zeros are plotted
    """

    # Convert the given system to a list if there is only one
    if type(sys_list) is not list:
        sys_list = [sys_list]

    # Create the figure to plot the traces
    if fig is None:
        fig = go.FigureWidget()
        # Define the figure layout
        fig.layout = go.Layout(
            title=title,
            xaxis=dict(title="Real Axis"),
            yaxis=dict(title="Imaginary Axis"),
            # template="plotly_white",
            legend=dict(groupclick="toggleitem"),
            showlegend=True,
        )
    else:  # Remove the previous plots if the figure already exist
        fig.data = []

    for sys_index, sys in enumerate(sys_list):
        poles = sys.pole()
        zeros = sys.zero()

        # Plot the locations of the poles and zeros
        if len(poles) > 0:
            trace_poles = go.Scatter(
                x=np.real(poles),
                y=np.imag(poles),
                mode="markers",
                marker=dict(
                    color=color_list[sys_index], size=10, symbol="x-thin", line_width=2, line_color=color_list[sys_index]
                ),
                name="Poles",
                legendgroup="sys{}".format(sys_index+1),
                legendgrouptitle_text="sys{}".format(sys_index+1),
            )
            fig.add_trace(trace_poles)

        if len(zeros) > 0:
            trace_zeros = go.Scatter(
                x=np.real(zeros),
                y=np.imag(zeros),
                mode="markers",
                marker=dict(
                    color="rgba(255, 255, 255, 0)",
                    size=10,
                    symbol="circle",
                    line_width=2,
                    line_color=color_list[sys_index],
                ),
                name="Zeros",
                legendgroup="sys{}".format(sys_index+1),
            )
            fig.add_trace(trace_zeros)

    # Show the figure
    if plot:
        fig.show()

    # Return the figure and the locations of poles and zeros as a tuple
    return fig


# %% Function stepWithInfo
def stepWithInfo(
    sys,
    info=None,
    T=None,
    SettlingTimeThreshold=0.05,
    RiseTimeThresholdMin=0.10,
    RiseTimeThresholdMax=0.90,
    resolution=10000,
    NameOfFigure="",
    sysName="",
    linestyle="-",
    plot_st=True,
    plot_rt=True,
    plot_overshoot=True,
    plot_DCGain=True,
):
    """
    Trace the step response and the interesting points and return those
    interesting informations.

    WARNING: Overshoot is in %!

    Parameters
    ----------
    sys: Linear Time Invariant (LTI)
        System analysed.

    info: Info, optional
        Object in which to store all the informations of the step response

    T: 1D array, optional
        Time vector.

    SettlingTimeThreshold: float, optional
        Threshold of the settling time.

    RiseTimeThresholdMin: float, optional
        Lower rise time threshold.

    RiseTimeThresholdMax: float, optional
        Upper rise time threshold.

    resolution: long, optional
        Number of points calculated to trace the step response.

    NameOfFigure: String, optional
        Name of the figure in which plotting the step response.

    sysName: String, optional
        Name of the system to plot.

    linestyle: '-.' , '--' , ':' , '-' , optional
        The line style used to plot the step response (default is '-').

    plot_st: bool, optionnal
        Plot the settling time point if True (default is True).

    plot_rt: bool, optionnal
        Plot the rise time point if True (default is True).

    plot_overshoot: bool, optionnal
        Plot the overshoot point if True (default is True).

    plot_DCGain: bool, optionnal
        Plot the DC gain point if True (default is True).

    Returns
    -------
    info: Info
        Object in which all the informations of the step response are stored.
    """

    [yout, t] = step_(sys, T, resolution, NameOfFigure, sysName, linestyle=linestyle)

    # Add the interestings points to the plot and store the informations in
    # info
    info = step_info(
        t,
        yout,
        info,
        SettlingTimeThreshold,
        RiseTimeThresholdMin,
        RiseTimeThresholdMax,
        plot_st,
        plot_rt,
        plot_overshoot,
        plot_DCGain,
    )

    return info


# %% Fonction pour tracer les résultats du step
def step_(sys, T=None, resolution=10000, NameOfFigure="", sysName="", linestyle="-"):
    """
    Trace the step with the given parameters.

    Parameters
    ----------
    sys: Linear Time Invariant (LTI)
        System analysed.

    T: 1D array, optional
        Time vector.

    resolution: long, optional
        Number of points calculated to trace the step response.

    NameOfFigure: String, optional
        Name of the figure in which plotting the step response.

    sysName: String, optional
        Name of the system to plot.

    linestyle: '-.' , '--' , ':' , '-' , optional
        The line style used to plot the step response (default is '-').

    Returns
    -------
    yout: 1D array
        Response of the system.

    t: 1D array
        Time vector.
    """

    if NameOfFigure == "":
        plt.figure()
    else:
        plt.figure(NameOfFigure)
    [yout, t] = ml.step(sys, T)
    # Pour modifier la résolution
    [yout, t] = ml.step(sys, np.linspace(t[0], t[-1], resolution))

    # Arrondi les valeurs à x décimales
    # yout = np.around(yout, decimals=6)
    # t = np.around(t, decimals=6)

    plt.plot(t, yout, label=sysName, linestyle=linestyle)
    plt.title("Step Response")
    plt.ylabel("Magnitude")
    plt.xlabel("Time (seconds)")
    if sysName != "":
        plt.legend()

    return [yout, t]


# %% Fonction pour tracer les résultats du step sur un graphe intéractif (=> JupyterBook)
def step(
    sys_list, T=None, resolution=10000, fig=None, title="Step response", data=False
):
    """
    Trace the step with the given parameters.

    Parameters
    ----------
    sys_list: system or list of systems)
        A single system or a list of systems to analyse

    T: 1D array, optional
        Time vector.

    resolution: long, optional
        Number of points calculated to trace the step response.

    fig: go.FigureWidget
        Figure where the step responses must be plotted.

    title: String, optional
        Title of the figure in which plotting the step response.

    data: bool, optionnal
        Return the plot data (yout and t) for all the systems plotted.

    Returns
    -------
    fig: plotly.graph_objs.FigureWidget
        Figure where the step responses are plotted.

    youts: 2D array
        Responses of each system.

    ts: 2D array
        Time vectors of each system.
    """

    # Convert the given system to a list if there is only one
    if type(sys_list) is not list:
        sys_list = [sys_list]

    # Create the figure to plot the traces
    if fig is None:
        fig = go.Figure()
        # Define the figure layout
        fig.layout = go.Layout(
            title=title,
            xaxis=dict(title="Time (seconds)"),
            yaxis=dict(title="Magnitude"),
            # template="plotly_white",
        )
    else:  # Remove the previous plots if the figure already exist
        fig.data = []

    if data:
        youts = [[] * resolution for i in range(len(sys_list))]
        ts = [[] * resolution for i in range(len(sys_list))]

    for sys_index, sys in enumerate(sys_list):
        [yout, t] = ml.step(sys, T)
        # Pour modifier la résolution
        [yout, t] = ml.step(sys, np.linspace(t[0], t[-1], resolution))

        # Generate the trace and plot it in the figure
        trace_step = go.Scatter(
            x=t,
            y=yout,
            name="sys{}".format(sys_index + 1),
            line=dict(color=color_list[sys_index]),
        )
        fig.add_trace(trace_step)

        if data:
            for i in range(resolution):
                youts[sys_index].append(yout[i])
                ts[sys_index].append(t[i])

    # Show the figure if the function is called at the and of the notebook cell
    fig

    if data:
        return [fig, youts, ts]
    else:
        return fig


def step2(
    sys_list, T=None, resolution=10000, fig=None, title="Step response", data=False
):
    """
    Trace the step with the given parameters.

    Parameters
    ----------
    sys_list: system or list of systems
        A single system or a list of systems to analyse

    T: 1D array, optional
        Time vector.

    resolution: long, optional
        Number of points calculated to trace the step response.

    fig: go.FigureWidget
        Figure where the step responses must be plotted.

    title: String, optional
        Title of the figure in which plotting the step response.

    data: bool, optionnal
        Return the plot data (yout and t) for all the systems plotted.

    Returns
    -------
    fig: plotly.graph_objs.FigureWidget
        Figure where the step responses are plotted.

    youts: 2D array
        Responses of each system.

    ts: 2D array
        Time vectors of each system.
    """

    # Convert the given system to a list if there is only one
    if type(sys_list) is not list:
        sys_list = [sys_list]

    # Create the figure to plot the traces
    if fig is None:
        fig = go.FigureWidget()
        # Define the figure layout
        fig.layout = go.Layout(
            title=title,
            xaxis=dict(title="Time (seconds)"),
            yaxis=dict(title="Magnitude"),
            # template="plotly_white",
        )
    else:  # Remove the previous plots if the figure already exist
        fig.data = []

    if data:
        youts = [[] * resolution for i in range(len(sys_list))]
        ts = [[] * resolution for i in range(len(sys_list))]

    # Plot the step response of each system
    for sys_index, sys in enumerate(sys_list):
        [yout, t] = ml.step(sys, T)
        # Pour modifier la résolution
        [yout, t] = ml.step(sys, np.linspace(t[0], t[-1], resolution))

        # Generate the trace and plot it in the figure
        trace_step = go.Scatter(
            x=t,
            y=yout,
            name="sys{}".format(sys_index + 1),
            line=dict(color=color_list[sys_index]),
        )
        fig.add_trace(trace_step)

        if data:
            for i in range(resolution):
                youts[sys_index].append(yout[i])
                ts[sys_index].append(t[i])

    if data:
        return [fig, youts, ts]
    else:
        return fig


# %% Fonction step_info
def step_info(
    t,
    yout,
    info=None,
    SettlingTimeThreshold=0.05,
    RiseTimeThresholdMin=0.10,
    RiseTimeThresholdMax=0.90,
    plot_st=True,
    plot_rt=True,
    plot_overshoot=True,
    plot_DCGain=True,
):
    """
    Trace the interesting points of a given step plot.

    Parameters
    ----------
    t: 1D array
        Time vector.

    y: 1D array
        Response of the system.

    info: Info, optional
        Object in which to store all the informations of the step response

    SettlingTimeThreshold: float, optional
        Threshold of the settling time (default is 0.05).

    RiseTimeThresholdMin: float, optional
        Lower rise time threshold (default is 0.10).

    RiseTimeThresholdMax: float, optional
        Upper rise time threshold (default is 0.90).

    plot_st: bool, optionnal
        Plot the settling time point if True (default is True).

    plot_rt: bool, optionnal
        Plot the rise time point if True (default is True).

    plot_overshoot: bool, optionnal
        Plot the overshoot point if True (default is True).

    plot_DCGain: bool, optionnal
        Plot the DC gain point if True (default is True).

    Returns
    -------
    info: Info
        Object in which all the informations of the step response are stored.
    """

    # Creation of the object info if not given
    if info is None:
        info = Info()

    # Store the colour of the current plot
    color = plt.gca().lines[-1].get_color()

    # Calcul du dépassement en prenant la valeur max retourné par step et en la
    # divisant par la valeur finale
    osIndice = np.where(yout == np.amax(yout))  # renvoie un tuple d'array
    osIndice = osIndice[-1][-1]  # lit le dernier indice répondant à la condition

    info.Peak = yout.max()
    info.Overshoot = (yout.max() / yout[-1] - 1) * 100
    info.PeakTime = float(t[osIndice])
    # print ("Overshoot:", info.Overshoot)

    if plot_overshoot:
        plt.plot(
            [t[osIndice], t[osIndice]], [0, yout[osIndice]], "k-.", linewidth=0.5
        )  # Vertical
        plt.plot(
            [t[0], t[osIndice]], [yout[osIndice], yout[osIndice]], "k-.", linewidth=0.5
        )  # Horizontale
        plt.plot(t[osIndice], yout[osIndice], color=color, marker="o")

    # Calcul du temps de montée en fonction du treshold (par défaut: de 10% à
    # 90% de la valeur finale)
    delta_values = yout[-1] - yout[0]
    RiseTimeThresholdMinIndice = next(
        i
        for i in range(0, len(yout) - 1)
        if yout[i] - yout[0] > delta_values * RiseTimeThresholdMin
    )
    RiseTimeThresholdMaxIndice = next(
        i
        for i in range(0, len(yout) - 1)
        if yout[i] - yout[0] > delta_values * RiseTimeThresholdMax
    )

    RiseTimeThreshold = [None] * 2
    RiseTimeThreshold[0] = t[RiseTimeThresholdMinIndice] - t[0]
    RiseTimeThreshold[1] = t[RiseTimeThresholdMaxIndice] - t[0]
    info.RiseTime = RiseTimeThreshold[1] - RiseTimeThreshold[0]
    # print ("RiseTime:", info.RiseTime)

    if plot_rt:
        plt.plot(
            [t[RiseTimeThresholdMinIndice], t[RiseTimeThresholdMinIndice]],
            [0, yout[RiseTimeThresholdMaxIndice]],
            "k-.",
            linewidth=0.5,
        )  # Limite gauche
        plt.plot(
            [t[RiseTimeThresholdMaxIndice], t[RiseTimeThresholdMaxIndice]],
            [0, yout[RiseTimeThresholdMaxIndice]],
            "k-.",
            linewidth=0.5,
        )  # Limite droite
        plt.plot(
            [t[0], t[RiseTimeThresholdMaxIndice]],
            [yout[RiseTimeThresholdMaxIndice], yout[RiseTimeThresholdMaxIndice]],
            "k-.",
            linewidth=0.5,
        )  # Limite horizontale
        plt.plot(
            t[RiseTimeThresholdMaxIndice],
            yout[RiseTimeThresholdMaxIndice],
            color=color,
            marker="o",
        )

    # Calcul du temps de réponse à x% (5% par défaut)
    settlingTimeIndice = next(
        i
        for i in range(len(yout) - 1, 1, -1)
        if abs(yout[i] - yout[0]) / delta_values > (1 + SettlingTimeThreshold)
        or abs(yout[i] - yout[0]) / delta_values < (1 - SettlingTimeThreshold)
    )
    info.SettlingTime = t[settlingTimeIndice] - t[0]
    # print ("SettlingTime:", info.SettlingTime)

    if plot_st:
        plt.plot(
            [0, max(t)],
            [
                yout[0] + delta_values * (1 + SettlingTimeThreshold),
                yout[0] + delta_values * (1 + SettlingTimeThreshold),
            ],
            "k-.",
            linewidth=0.5,
        )  # Limite haute
        plt.plot(
            [0, max(t)],
            [
                yout[0] + delta_values * (1 - SettlingTimeThreshold),
                yout[0] + delta_values * (1 - SettlingTimeThreshold),
            ],
            "k-.",
            linewidth=0.5,
        )  # Limite basse
        plt.plot(
            [t[settlingTimeIndice], t[settlingTimeIndice]],
            [0, yout[settlingTimeIndice]],
            "k-.",
            linewidth=0.5,
        )  # Vertical
        plt.plot(
            t[settlingTimeIndice], yout[settlingTimeIndice], color=color, marker="o"
        )

    # Gain statique
    info.DCGain = yout[-1]
    if plot_DCGain:
        plt.plot([0, max(t)], [yout[-1], yout[-1]], "k:", linewidth=0.5)
        plt.plot(t[-1], yout[-1], color=color, marker="o")
    # print ("DC gain:", info.DCGain)

    return info


# %% Step from a value to another


def stepFromTo(
    sys, value_init, value_fin, resolution=10000, focus=True, NameOfFigure=""
):
    """
    Trace the step when the input goes from a given value to another.

    Parameters
    ----------
    sys: Linear Time Invariant (LTI)
        System analysed.

    value_init: float
        Initial value of the intput.

    value_fin: float
        Final value of the intput.

    resolution: long, optional
        Number of points calculated to trace the step response.

    focus: boolean, optional
        Plot the interresting part of the step as the standard step function (True by default).

    NameOfFigure: String, optional
        Name of the figure in which plotting the step response.

    Returns
    -------
    Peak: float
        Peak absolute value of y(t).

    PeakTime: float
        Time at which the peak value occurs.

    yout_useful: 1D array
        Response of the system for the interresting part of the step.

    t_useful: 1D array
        Time vector of the interresting part of the step.
    """

    from scipy import signal

    __, t = ml.step(sys)  # To get an optimized siez of t
    T = (
        t[-1] * 2
    )  # To double the size of t to be able to trace the step from 0 and after stabilisation from the initial value

    t = np.linspace(0, T, resolution + 1)  # resolution+1 because t begins at 0
    delta_values = value_fin - value_init
    moy_values = (value_fin + value_init) / 2
    sq = (
        -delta_values / 2 * signal.square(2 * np.pi * (1 / T) * t) + moy_values
    )  # '-' to start from the bottom
    sq[0] = 0  # To start from 0 as the output of the lsim function start from 0
    sq[-1] = value_fin  # To avoid to come back to the inital value

    yout, t, __ = ml.lsim(sys, sq, t)  # Calculus of the special step

    yout_useful = yout[int(resolution / 2) :]  # To limit the research after the step

    peak = np.amax(yout_useful)
    peak_indice = np.where(yout_useful == np.amax(yout_useful))
    peak_indice = peak_indice[-1][-1]
    peak_time = peak_indice * (T / resolution)

    t_useful = np.linspace(
        0, T / 2, int(resolution / 2) + 1
    )  # Calculus of the time vector to focus on the interresting part of the step

    if NameOfFigure == "":
        plt.figure("Step " + str(value_init) + " -> " + str(value_fin))
    else:
        plt.figure(NameOfFigure)
    if focus:
        plt.plot(t_useful, yout_useful)
    else:
        plt.plot(t, sq)  # Plot the input
        plt.plot(t, yout)  # Plot the output
        plt.legend(["U(t)", "Y(t)"])
        plt.plot(T / 2 + peak_time, peak, "ko")

    return peak, peak_time, t_useful, yout_useful


# %% Get the gain and the frequency at a given phase
def getValues(
    sys, phaseValue, mag=None, phase=None, omega=None, printValue=True, NameOfFigure=""
):
    """
    Get the values of the gain and the frequency at a given phase of the
    system.

    Get the values of the gain and the frequency at a given phase from given
    arrays of gains, phases and frequencies.

    Parameters
    ----------
    sys: Linear Time Invariant (LTI)
        System analysed.

    phaseValue: float
        Phase at which we want to get the gain and frequency values.

    mag: 1D array, optional
        Array of gains (not in dB).

    phase: 1D array, optional
        Array of phases.

    omega: 1D array, optional
        Array of frequencies (in radians).

    printValue: boolean, optional
        Print values if True (by default).

    NameOfFigure: String, optional
        Name of the figure in which to plot.

    Returns
    -------
    mag: float
        The gain value for the given phase.

    omega: float
        The frequency value in rad/sec for the given phase.
    """

    lowLimit = -2
    highLimit = 2
    if NameOfFigure == "":
        plt.figure()
    else:
        plt.figure(NameOfFigure)

    if np.all(mag is None) and np.all(phase is None) and np.all(omega is None):
        # liste de fréquences afin d'augmenter la résolution de calcul (par défaut: 50 éléments)
        w = np.logspace(lowLimit, highLimit, 10000)
        mag, phase, omega = ml.bode(sys, w, dB=True, Hz=False, deg=True)
        phase = (
            phase * 180 / math.pi
        )  # Pour avoir les phases en degrés plutôt qu'en radians
        idx = (np.abs(phase - phaseValue)).argmin()
        while idx in (np.size(phase) - 1, 0):
            if idx == 0:
                lowLimit -= 1
            else:
                highLimit += 1
            # liste de fréquences afin d'augmenter la résolution de calcul (par défaut: 50 éléments)
            w = np.logspace(lowLimit, highLimit, 10000)
            mag, phase, omega = ml.bode(sys, w, dB=True, Hz=False, deg=True)
            phase = (
                phase * 180 / math.pi
            )  # Pour avoir les phases en degrés plutôt qu'en radians
            idx = (np.abs(phase - phaseValue)).argmin()

    else:
        phase = (
            phase * 180 / math.pi
        )  # Pour avoir les phases en degrés plutôt qu'en radians
        idx = (np.abs(phase - phaseValue)).argmin()

    if printValue:
        mag_dB = 20 * np.log10(mag[idx])  # Pour avoir les gains en dB
        print(f"Gain à {phaseValue}° = {mag_dB} dB")
        print(f"Fréquence à {phaseValue}° = {omega[idx]} rad/sec")

    return mag[idx], omega[idx]


# %% Compute reasonable defaults for axes
def default_frequency_range(
    syslist, Hz=None, number_of_samples=None, feature_periphery_decades=None
):
    """Compute a reasonable default frequency range for frequency
    domain plots.

    Finds a reasonable default frequency range by examining the features
    (poles and zeros) of the systems in syslist.

    Parameters
    ----------
    syslist : list of LTI
        List of linear input/output systems (single system is OK)
    Hz : bool
        If True, the limits (first and last value) of the frequencies
        are set to full decades in Hz so it fits plotting with logarithmic
        scale in Hz otherwise in rad/s. Omega is always returned in rad/sec.
    number_of_samples : int, optional
        Number of samples to generate.  The default value is read from
        ``config.defaults['freqplot.number_of_samples'].  If None, then the
        default from `numpy.logspace` is used.
    feature_periphery_decades : float, optional
        Defines how many decades shall be included in the frequency range on
        both sides of features (poles, zeros).  The default value is read from
        ``config.defaults['freqplot.feature_periphery_decades']``.

    Returns
    -------
    omega : array
        Range of frequencies in rad/sec

    Examples
    --------
    >>> from matlab import ss
    >>> sys = ss("1. -2; 3. -4", "5.; 7", "6. 8", "9.")
    >>> omega = _default_frequency_range(sys)

    """
    # This code looks at the poles and zeros of all of the systems that
    # we are plotting and sets the frequency range to be one decade above
    # and below the min and max feature frequencies, rounded to the nearest
    # integer.  It excludes poles and zeros at the origin.  If no features
    # are found, it turns logspace(-1, 1)

    # Set default values for options
    number_of_samples = 1000
    feature_periphery_decades = 0.1

    # Find the list of all poles and zeros in the systems
    features = np.array(())
    freq_interesting = []

    # detect if single sys passed by checking if it is sequence-like
    if not hasattr(syslist, "__iter__"):
        syslist = (syslist,)

    for sys in syslist:
        try:
            # Add new features to the list
            if sys.isctime():
                features_ = np.concatenate((np.abs(sys.pole()), np.abs(sys.zero())))
                # Get rid of poles and zeros at the origin
                features_ = features_[features_ != 0.0]
                features = np.concatenate((features, features_))
            elif sys.isdtime(strict=True):
                fn = math.pi * 1.0 / sys.dt
                # TODO: What distance to the Nyquist frequency is appropriate?
                freq_interesting.append(fn * 0.9)

                features_ = np.concatenate((sys.pole(), sys.zero()))
                # Get rid of poles and zeros
                # * at the origin and real <= 0 & imag==0: log!
                # * at 1.: would result in omega=0. (logaritmic plot!)
                features_ = features_[(features_.imag != 0.0) | (features_.real > 0.0)]
                features_ = features_[
                    np.bitwise_not(
                        (features_.imag == 0.0)
                        & (np.abs(features_.real - 1.0) < 1.0e-10)
                    )
                ]
                # TODO: improve
                features__ = np.abs(np.log(features_) / (1.0j * sys.dt))
                features = np.concatenate((features, features__))
            else:
                # TODO
                raise NotImplementedError("type of system in not implemented now")
        except NotImplementedError:
            pass

    # Make sure there is at least one point in the range
    if features.shape[0] == 0:
        features = np.array([1.0])

    if Hz:
        features /= 2.0 * math.pi
        features = np.log10(features)
        lsp_min = np.floor(np.min(features) - feature_periphery_decades)
        lsp_max = np.ceil(np.max(features) + feature_periphery_decades)
        lsp_min += np.log10(2.0 * math.pi)
        lsp_max += np.log10(2.0 * math.pi)
    else:
        features = np.log10(features)
        lsp_min = np.floor(np.min(features) - feature_periphery_decades)
        lsp_max = np.ceil(np.max(features) + feature_periphery_decades)
    if freq_interesting:
        lsp_min = min(lsp_min, np.log10(min(freq_interesting)))
        lsp_max = max(lsp_max, np.log10(max(freq_interesting)))

    # TODO: Add a check in discrete case to make sure we don't get aliasing
    # (Attention: there is a list of system but only one omega vector)

    # Set the range to be an order of magnitude beyond any features
    if number_of_samples:
        omega = np.logspace(lsp_min, lsp_max, num=number_of_samples, endpoint=True)
    else:
        omega = np.logspace(lsp_min, lsp_max, endpoint=True)
    return omega


# %% Utility function to unwrap an angle measurement
def unwrap(angle, period=2 * math.pi):
    """Unwrap a phase angle to give a continuous curve

    Parameters
    ----------
    angle : array_like
        Array of angles to be unwrapped
    period : float, optional
        Period (defaults to ``2*pi``)

    Returns
    -------
    angle_out : array_like
        Output array, with jumps of period/2 eliminated

    Examples
    --------
    >>> import numpy as np
    >>> theta = [5.74, 5.97, 6.19, 0.13, 0.35, 0.57]
    >>> unwrap(theta, period=2 * np.pi)
    [5.74, 5.97, 6.19, 6.413185307179586, 6.633185307179586, 6.8531853071795865]

    """
    dangle = np.diff(angle)
    dangle_desired = (dangle + period / 2.0) % period - period / 2.0
    correction = np.cumsum(dangle_desired - dangle)
    angle[1:] += correction
    return angle

    # %% Utility function to unwrap an angle measurement

    # def unwrap(angle, period=2*math.pi):
    """Unwrap a phase angle to give a continuous curve

    Parameters
    ----------
    angle : array_like
        Array of angles to be unwrapped
    period : float, optional
        Period (defaults to `2*pi`)

    Returns
    -------
    angle_out : array_like
        Output array, with jumps of period/2 eliminated

    Examples
    --------
    >>> import numpy as np
    >>> theta = [5.74, 5.97, 6.19, 0.13, 0.35, 0.57]
    >>> unwrap(theta, period=2 * np.pi)
    [5.74, 5.97, 6.19, 6.413185307179586, 6.633185307179586, 6.8531853071795865]

    """
    dangle = np.diff(angle)
    dangle_desired = (dangle + period / 2.0) % period - period / 2.0
    correction = np.cumsum(dangle_desired - dangle)
    angle[1:] += correction
    return angle


# %% Interactive Bode plot based on the package control_plotly
def bode(sys_list, margins=False):
    """Plot an interactive Bode diagram

    Args:
        sys_list (system or list of systems): A single system or a list of systems to analyse
        margins (bool, optional): Plot margins (phase and gain margins) if True. Defaults to False.

    Returns:
        plotly.graph_objs.Figure: The figure where the Bode diagram is plotted
    """
    # Convert the given system to a list if there is only one
    if type(sys_list) is not list:
        sys_list = [sys_list]

    fig = cp.bode(sys_list)
    fig.update_layout(
        title="Bode Plot",
        #template="plotly_white",
        # height=600, width=1000,
        showlegend=True,
    )

    
    visible = True  # To manage the visibility of the traces
    for sys_index, sys in enumerate(sys_list):
        # To group the two traces by sys in a same group and to show it only one time in the legend
        fig.data[sys_index].showlegend = True
        fig.data[sys_index].legendgroup = "sys{}".format(sys_index)
        fig.data[sys_index]["line"] = dict(color=color_list[sys_index + 1])
        fig.data[sys_index + len(sys_list)]["line"] = dict(color=color_list[sys_index + 1])
        fig.data[sys_index + len(sys_list)].legendgroup = "sys{}".format(sys_index)

        if sys_index > 0:
            visible = "legendonly"  # To avoid to have to much traces on the figure by default

        if margins:
            gm, pm, wcg, wcp = ct.margin(
                sys
            )  # Extract the gain (Gm) and phase (Pm) margins
            gm = 20 * np.log10(gm)  # Conversion of gm in dB

            # Figure out sign of the phase at the first gain crossing (needed if phase_wrap is True)
            mag, phase, omega = ct.bode(sys, plot=False)
            phase_at_cp = phase[(np.abs(omega - wcp)).argmin()]
            if phase_at_cp >= 0.0:
                phase_limit = 180.0
            else:
                phase_limit = -180.0

            # Figure out what are the limits of the figure
            x_mins, x_maxs, y_mins, y_maxs = figure_limits(fig, 2)

            # Draw a line at gain limit
            fig.add_traces(
                go.Scatter(
                    x=[x_mins[0], x_maxs[0]],
                    y=[0, 0],
                    mode="lines",
                    line=dict(color="black", width=2, dash="dot"),
                    name="Margins - sys{}".format(sys_index + 1),
                    legendgroup="Margins - sys{}".format(sys_index + 1),
                    showlegend=False,
                    hoverinfo="none",
                    visible=visible,
                ),
                rows=1,
                cols=1,
            )
            # Draw a line at phase limit
            fig.add_traces(
                go.Scatter(
                    x=[x_mins[1], x_maxs[1]],
                    y=[phase_limit, phase_limit],
                    mode="lines",
                    line=dict(color="black", width=2, dash="dot"),
                    name="Margins - sys{}".format(sys_index + 1),
                    legendgroup="Margins - sys{}".format(sys_index + 1),
                    showlegend=False,
                    hoverinfo="none",
                    visible=visible,
                ),
                rows=2,
                cols=1,
            )

            # Annotate the phase margin (if it exists)
            if pm != float("inf") and wcp != float("nan"):
                fig.add_traces(
                    go.Scatter(
                        x=[wcp, wcp],
                        y=[0, y_mins[0]],
                        mode="lines",
                        line=dict(color="black", width=2, dash="dot"),
                        legendgroup="Margins - sys{}".format(sys_index + 1),
                        showlegend=False,
                        hoverinfo="none",
                        visible=visible,
                    ),
                    rows=1,
                    cols=1,
                )

                fig.add_traces(
                    go.Scatter(
                        x=[wcp, wcp],
                        y=[phase_limit, y_maxs[1]],
                        mode="lines",
                        line=dict(color="black", width=2, dash="dot"),
                        legendgroup="Margins - sys{}".format(sys_index + 1),
                        showlegend=False,
                        hoverinfo="none",
                        visible=visible,
                    ),
                    rows=2,
                    cols=1,
                )

            # Annotate the gain margin (if it exists)
            if gm != float("inf") and wcg != float("nan"):
                fig.add_traces(
                    go.Scatter(
                        x=[wcg, wcg],
                        y=[phase_limit, y_maxs[1]],
                        mode="lines",
                        line=dict(color="black", width=2, dash="dot"),
                        legendgroup="Margins - sys{}".format(sys_index + 1),
                        showlegend=False,
                        hoverinfo="none",
                        visible=visible,
                    ),
                    rows=2,
                    cols=1,
                )

                fig.add_traces(
                    go.Scatter(
                        x=[wcg, wcg],
                        y=[0, y_mins[0]],
                        mode="lines",
                        line=dict(color="black", width=2, dash="dot"),
                        legendgroup="Margins - sys{}".format(sys_index + 1),
                        showlegend=False,
                        hoverinfo="none",
                        visible=visible,
                    ),
                    rows=1,
                    cols=1,
                )

            # Plot the gain margin
            fig.add_traces(
                go.Scatter(
                    x=[wcg, wcg],
                    y=[0, -gm],
                    mode="lines",
                    line=dict(color="black", width=2, dash="solid"),
                    name="Margins - sys{}".format(sys_index + 1),
                    legendgroup="Margins - sys{}".format(sys_index + 1),
                    showlegend=False,
                    hoverinfo="none",
                    visible=visible,
                ),
                rows=1,
                cols=1,
            )
            fig.add_traces(
                go.Scatter(
                    x=[wcg],
                    y=[-gm],
                    mode="markers",
                    marker=dict(
                        color="black",
                        size=6,
                        symbol="circle",
                        line_width=2,
                        line_color="black",
                    ),
                    name="sys{}".format(sys_index + 1),
                    legendgroup="Margins - sys{}".format(sys_index + 1),
                    hoverinfo="none",
                    customdata=[gm],
                    hovertemplate="<b>Gain margin</b>: %{customdata:.3f} dB",  # The <extra></extra> balise is used to avoid to print the trace name on hover
                    showlegend=False,
                    visible=visible,
                ),
                rows=1,
                cols=1,
            )

            # Plot the phase margin
            fig.add_traces(
                go.Scatter(
                    x=[wcp, wcp],
                    y=[phase_limit, phase_limit + pm],
                    mode="lines",
                    line=dict(color="black", width=2, dash="solid"),
                    name="Margins - sys{}".format(sys_index + 1),
                    legendgroup="Margins - sys{}".format(sys_index + 1),
                    showlegend=False,
                    hoverinfo="none",
                    visible=visible,
                ),
                rows=2,
                cols=1,
            )
            fig.add_traces(
                go.Scatter(
                    x=[wcp],
                    y=[phase_limit + pm],
                    mode="markers",
                    marker=dict(
                        color="black",
                        size=6,
                        symbol="circle",
                        line_width=2,
                        line_color="black",
                    ),
                    name="sys{}".format(sys_index + 1),
                    legendgroup="Margins - sys{}".format(sys_index + 1),
                    hoverinfo="none",
                    customdata=[pm],
                    hovertemplate="<b>Phase margin</b>: %{customdata:.3f}°",  # The <extra></extra> balise is used to avoird to print the trace name on hover
                    showlegend=True,
                    visible=visible,
                ),
                rows=2,
                cols=1,
            )

    # Show the figure if the function is called at the and of the notebook cell
    fig

    return fig

# Interactive Bode plot with widgets compatibility (Works only with Jupyter Notebooks !)
def bode2(sys_list, fig=None):
    """Plot the Bode diagram based on a given system or a list of given systems

    Args:
        sys_list (system or list of systems): A single system or a list of systems to analyse
        fig (plotly.graph_objs.FigureWidget): The figure where to plots the poles and zeros

    Returns:
        plotly.graph_objs.FigureWidget: The figure where the Bode diagram is plotted
    """

    # Convert the given system to a list if there is only one
    if type(sys_list) is not list:
        sys_list = [sys_list]

    if fig is None:
        fig_init = bode(sys_list)
        fig = go.FigureWidget(fig_init)
    else:  # Updating of existing traces
        for sys_index, sys in enumerate(sys_list):
            w = np.logspace(-2, 2, 10000)
            mag_list, phase_list, w = ct.bode_plot(
                sys,
                omega=w,
                dB=True,
                plot=False,
                omega_limits=None,
                omega_num=None,
                margins=None,
            )
            mag_list = 20*np.log10(mag_list)
            phase_list = phase_list*180/(np.pi)

            # Gain plot
            fig.data[sys_index].x = w
            fig.data[sys_index].y = mag_list

            # Phase plot
            fig.data[sys_index + len(sys_list)].x = w
            fig.data[sys_index + len(sys_list)].y = phase_list

    return fig

# %% Function to plot Nichols diagram as needed for the laboratory
# Static Nichols function reviewed by Alban Van Laethem based on matplotlib
def nichols_(
    sys_list,
    omega=None,
    grid=None,
    labels=[""],
    NameOfFigure="",
    data=False,
    ax=None,
    linestyle="-",
):
    """Nichols plot for a system

    Plots a Nichols plot for the system over a (optional) frequency range.

    Parameters
    ----------
    sys_list: list of LTI, or LTI
        List of linear input/output systems

    omega: array_like, optional
        Range of frequencies (list or bounds) in rad/sec

    grid: boolean, optional
        True if the plot should include a Nichols-chart grid. Default is True.

    labels: list of Strings
        List of the names of the given systems

    NameOfFigure: String, optional
        Name of the figure in which to plot.

    data: boolean, optional
        True if we must return x and y (default is False)

    ax: axes.subplots.AxesSubplot, optional
        The axe on which to plot

    linestyle: '-.' , '--' , ':' , '-' , optional
        The line style used to plot the nichols graph (default is '-').

    Returns
    -------
    if data == True:
        x: 1D array
            Abscisse vector
        y: 1D array
            Ordinate vector
    """

    # Open a figure with the given name or open a new one
    if NameOfFigure == "":
        plt.figure()
    else:
        plt.figure(NameOfFigure)

    ax = ax or plt.gca()

    # Get parameter values
    # grid = config._get_param('nichols', 'grid', grid, True)

    # If argument was a singleton, turn it into a list
    if not getattr(sys_list, "__iter__", False):
        sys_list = (sys_list,)

    # Select a default range if none is provided
    if omega is None:
        omega = default_frequency_range(sys_list)

    for index, sys in enumerate(sys_list):
        # Get the magnitude and phase of the system
        mag_tmp, phase_tmp, omega = sys.frequency_response(omega)
        mag = np.squeeze(mag_tmp)
        phase = np.squeeze(phase_tmp)

        # Convert to Nichols-plot format (phase in degrees,
        # and magnitude in dB)
        x = unwrap(np.degrees(phase), 360)
        y = 20 * np.log10(mag)

        # Generate the plot
        if labels != [""]:
            ax.plot(x, y, label=labels[index], linestyle=linestyle)
        else:
            ax.plot(x, y, linestyle=linestyle)

    ax.set_xlabel("Phase (deg)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Nichols Plot")

    # Mark the -180 point
    if x[-1] < -90:
        ax.plot([-180], [0], "r+", label="_nolegend_")

    # Add grid
    if grid:
        nichols_grid()

    # Add legend
    if labels != [""]:
        plt.legend()

    if data:
        return x, y

# Interactive Nichols plot based on the package control_plotly
def nichols(sys_list, show_grid=True, margins=False):
    """Nichols plot for a system

    Plots a Nichols plot for the system over a (optional) frequency range.

    Args:
        sys_list (system or list of systems): A single system or a list of systems to analyse
        show_grid (bool, optional): Add the nichols grid. Defaults to True.
        margins (bool, optional): Find the margins and plot them. Defaults to False.

    Returns:
        plotly figure: plotly figure
    """

    # Convert the given system to a list if there is only one
    if type(sys_list) is not list:
        sys_list = [sys_list]

    w = np.logspace(-2, 2, 10000)
    fig = cp.nichols(sys_list, w=w, show_grid=show_grid)
    fig.update_traces(
        visible=False, selector=dict(mode="markers")
    )  # To hide the automatic critical point

    # To be able to toggle the visibility of the grid via the legend
    for trace_index, trace in enumerate(fig.data):
        if trace_index > len(sys_list)-1 and trace_index != len(fig.data)-1:
            trace.legendgroup="Grid"
            if trace_index == len(sys_list):
                trace.name='Grid'
                trace.showlegend=True
                trace.hoverinfo='text'
                trace.text='6.00 dB'

    # Mark the critical point (-180, 0)
    fig.add_trace(
        go.Scatter(
            x=[-180],
            y=[0],
            mode="markers",
            marker=dict(
                color="red",
                size=10,
                symbol="cross-thin",
                line_width=2,
                line_color="red",
            ),
            hoverinfo="none",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Nichols Plot",
        # template="plotly_white",
        # height=600,
        # width=1000,
        showlegend=True,
    )

    
    visible = True  # To manage the visibility of the traces
    for sys_index, sys in enumerate(sys_list):
        # To show the system in the legend
        fig.data[sys_index].showlegend = True
        # fig.data[sys_index].legendgroup='sys{}'.format(sys_index+1)
        # fig.data[sys_index+len(sys_list)].legendgroup='sys{}'.format(sys_index)

        if sys_index > 0:
            visible = "legendonly"  # To avoid to have to much traces on the figure by default

        if margins:

            gm, pm, wcg, wcp = ml.margin(
                sys
            )  # Extract the gain (Gm) and phase (Pm) margins
            gm = 20 * np.log10(gm)  # Conversion of gm in dB

            # Phase margin
            fig.add_trace(
                go.Scatter(
                    x=[-180, -180 + pm],
                    y=[0, 0],
                    mode="lines",
                    line=dict(color="black", width=2, dash="solid"),
                    name="Margins - sys{}".format(sys_index + 1),
                    legendgroup="Margins - sys{}".format(sys_index + 1),
                    showlegend=False,
                    visible=visible,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[-180 + pm],
                    y=[0],
                    mode="markers",
                    marker=dict(
                        color="black",
                        size=6,
                        symbol="circle",
                        line_width=2,
                        line_color="black",
                    ),
                    name="sys{}".format(sys_index + 1),
                    legendgroup="Margins - sys{}".format(sys_index + 1),
                    hoverinfo="none",
                    customdata=[pm],
                    hovertemplate="<b>Phase margin</b>: %{customdata:.3f}°",  # The <extra></extra> balise is used to avoird to print the trace name on hover
                    showlegend=False,
                    visible=visible,
                ),
            )

            # Gain margin
            fig.add_trace(
                go.Scatter(
                    x=[-180, -180],
                    y=[0, -gm],
                    mode="lines",
                    line=dict(color="black", width=2, dash="solid"),
                    name="Margins - sys{}".format(sys_index + 1),
                    legendgroup="Margins - sys{}".format(sys_index + 1),
                    showlegend=False,
                    visible=visible,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[-180],
                    y=[-gm],
                    mode="markers",
                    marker=dict(
                        color="black",
                        size=6,
                        symbol="circle",
                        line_width=2,
                        line_color="black",
                    ),
                    name="sys{}".format(sys_index + 1),
                    legendgroup="Margins - sys{}".format(sys_index + 1),
                    hoverinfo="none",
                    customdata=[gm],
                    hovertemplate="<b>Gain margin</b>: %{customdata:.3f} dB",  # The <extra></extra> balise is used to avoird to print the trace name on hover
                    showlegend=True,
                    visible=visible,
                ),
            )

    # Show the figure if the function is called at the and of the notebook cell
    fig

    return fig

# Interactive nichols plot with widgets compatibility (Works only with Jupyter Notebooks !)
def nichols2(sys_list, fig=None):
    """Plot the Nichols diagram based on a given system or a list of given systems

    Args:
        sys_list (system or list of systems): A single system or a list of systems to analyse
        fig (plotly.graph_objs.FigureWidget): The figure where to plots the poles and zeros

    Returns:
        plotly.graph_objs.FigureWidget: The figure where the Nichols diagram is plotted
    """
    # Convert the given system to a list if there is only one
    if type(sys_list) is not list:
        sys_list = [sys_list]

    if fig is None:
        fig_init = nichols(sys_list)
        fig = go.FigureWidget(fig_init)

        # To be able to toggle the visibility of the grid via the legend
        for trace_index, trace in enumerate(fig.data):
            if trace_index > len(sys_list)-1 and trace_index != len(fig.data)-1:
                trace.legendgroup="Grid"
                if trace_index == len(sys_list):
                    trace.name='Grid'
                    trace.showlegend=True
                    trace.hoverinfo='text'
                    trace.text='6.00 dB'

    else:  # Updating of existing traces
        for sys_index, sys in enumerate(sys_list):
            w = np.logspace(-2, 2, 10000)
            mag_list, phase_list, w = ct.bode_plot(
                sys,
                omega=w,
                dB=True,
                plot=False,
                omega_limits=None,
                omega_num=None,
                margins=None,
            )
            mag_list = 20*np.log10(mag_list)
            phase_list = phase_list*180/(np.pi)
            
            fig.data[sys_index].x = phase_list
            fig.data[sys_index].y = mag_list

    return fig

# %% Finding the border limits of a figure
def figure_limits(fig, nb_subplots=1):
    """Function to get the limit borders of a figure.

    Args:
        fig (plotly figure): The figure to analyse
        nb_subplots (int, optional): Number of subplots in the figure. Defaults to 1.

    Returns:
        floats[]: values of the border limits for each subplot : x_mins, x_maxs, y_mins, y_maxs
    """
    cnt = 0
    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []

    # Loop until having found the limits of all the subplots
    while nb_subplots - cnt > 0:
        # Retrieve data from each subplot
        for i in range(0, nb_subplots):
            subplot_data = fig["data"][i]
            x_data = subplot_data["x"]
            y_data = subplot_data["y"]
            # x limits
            x_mins.append(min(x_data))
            x_maxs.append(max(x_data))
            # y limits
            y_mins.append(min(y_data))
            y_maxs.append(max(y_data))

        # Incrementing the counter
        cnt += 1

    return x_mins, x_maxs, y_mins, y_maxs


# %% Interactive Nyquist plot
def nyquist(sys_list):
    """Plot the Nyquist diagram based on a given system or a list of given systems

    Args:
        sys_list (system or list of systems): A single system or a list of systems to analyse

    Returns:
        plotly.graph_objs.Figure: The figure where the Nyquist diagram is plotted
    """

    # Parameter for the figure limits
    max_limit = 0

    # Convert the given system to a list if there is only one
    if type(sys_list) is not list:
        sys_list = [sys_list]

    # Creation of the figure
    fig = go.Figure()

    # Calcul of the frequency response
    w = np.logspace(-2, 2, 10000)
    for sys_index, sys in enumerate(sys_list):
        real, imag, freq = ml.nyquist(sys, w, plot=False)
        mag = []
        phase = []
        customdata = []
        for i in range(len(real)):
            mag.append(np.sqrt(real[i] ** 2 + imag[i] ** 2))
            phase.append(np.arctan2(imag[i], real[i]) * 180 / np.pi)
            customdata.append((mag[i], phase[i]))

        # Plotting of the Nyquist diagram
        fig.add_trace(
            go.Scatter(
                x=real,
                y=imag,
                mode="lines",
                line=dict(color=color_list[sys_index]),
                name="sys{}".format(sys_index + 1),
                text=freq,
                customdata=customdata,
                hovertemplate="<b>w</b>: %{text:.3f} rad/s<br><b>mag</b>: %{customdata[0]:.3f}<br><b>phase</b>: %{customdata[1]:.3f}°<br><b>imag</b>: %{y:.3f}<br><b>real</b>: %{x:.3f}<br>",
                legendgroup="sys{}".format(sys_index + 1),
            )
        )

        # Find where to put the arrow
        imag_min_index = np.argmin(imag)
        imag_max_index = np.argmax(imag)
        if np.absolute(imag[imag_max_index]) - np.absolute(imag[imag_min_index]) < 0:
            index = imag_min_index
        else:
            index = imag_max_index

        y_arrow = imag[index]  # Imaginary part
        x_arrow = real[
            index
        ]  # Real part corresponding to the minimal/maximal imaginary part

        # Find the direction of the arrow
        if real[index] - real[index - 1] < 0:
            symbol = "triangle-left"
        else:
            symbol = "triangle-right"

        # Put the arrow head
        fig.add_trace(
            go.Scatter(
                x=[x_arrow],
                y=[y_arrow],
                mode="markers",
                marker=dict(
                    # color=fig.data[sys_index].line.color,
                    color=color_list[sys_index],
                    size=10,
                    symbol=symbol,
                    line_width=2,
                    # line_color=fig.data[sys_index].line.color,
                    line_color=color_list[sys_index],
                ),
                hoverinfo="none",
                showlegend=False,
                legendgroup="sys{}".format(sys_index + 1),
            )
        )

        # Refresh (if needed) the max limit point to have the best zoom for the figure
        if max_limit < np.absolute(imag[index]):
            max_limit = np.absolute(imag[index])
        if max_limit < np.absolute(np.amax(real)):
            max_limit = np.absolute(np.amax(real))
        if max_limit < np.absolute(np.amin(real)):
            max_limit = np.absolute(np.amin(real))

    # Mark the critical point (-1, 0)
    fig.add_trace(
        go.Scatter(
            x=[-1],
            y=[0],
            mode="markers",
            marker=dict(
                color="red",
                size=10,
                symbol="cross-thin",
                line_width=2,
                line_color="red",
            ),
            name="Critical point",
            hoverinfo="none",
            showlegend=False,
        )
    )

    # Formatting of the figure
    fig.update_xaxes(title_text="Real", range=[-max_limit, max_limit])
    fig.update_yaxes(title_text="Imaginary", range=[-max_limit, max_limit])
    fig.update_layout(
        title="Nyquist plot", height=800, width=800
    )

    # Show the figure if the function is called at the and of the notebook cell
    fig

    return fig


# Interactive Nyquist plot with widgets compatibility (Works only with Jupyter Notebooks !)
def nyquist2(sys_list, fig=None):
    """Plot the Nyquist diagram based on a given system or a list of given systems

    Args:
        sys_list (system or list of systems): A single system or a list of systems to analyse
        fig (plotly.graph_objs.FigureWidget): The figure where to plots the poles and zeros

    Returns:
        plotly.graph_objs.FigureWidget: The figure where the Nyquist diagram is plotted
    """

    # Parameter for the figure limits
    max_limit = 0

    # Convert the given system to a list if there is only one
    if type(sys_list) is not list:
        sys_list = [sys_list]

    # # List of the Plotly (D3) standard colors
    # color_list = [
    #     "#1F77B4",
    #     "#FF7F0E",
    #     "#2CA02C",
    #     "#D62728",
    #     "#9467BD",
    #     "#8C564B",
    #     "#E377C2",
    #     "#7F7F7F",
    #     "#BCBD22",
    #     "#17BECF",
    # ]

    # Create the figure to plot the traces
    if fig is None:
        fig = go.FigureWidget()
        # Define the figure layout
        fig.update_layout(
            title="Nyquist plot",
            # template="plotly_white",
            #   height=800, width=800
        )
    else:  # Remove the previous plots if the figure already exist
        fig.data = []

    # Calcul of the frequency response
    w = np.logspace(-2, 2, 10000)
    for sys_index, sys in enumerate(sys_list):
        real, imag, freq = ml.nyquist(sys, w, plot=False)
        mag = []
        phase = []
        customdata = []
        for i in range(len(real)):
            mag.append(np.sqrt(real[i] ** 2 + imag[i] ** 2))
            phase.append(np.arctan2(imag[i], real[i]) * 180 / np.pi)
            customdata.append((mag[i], phase[i]))

        # Plotting of the Nyquist diagram
        fig.add_trace(
            go.Scatter(
                x=real,
                y=imag,
                mode="lines",
                line=dict(color=color_list[sys_index]),
                name="sys{}".format(sys_index + 1),
                text=freq,
                customdata=customdata,
                hovertemplate="<b>w</b>: %{text:.3f} rad/s<br><b>mag</b>: %{customdata[0]:.3f}<br><b>phase</b>: %{customdata[1]:.3f}°<br><b>imag</b>: %{y:.3f}<br><b>real</b>: %{x:.3f}<br>",
                legendgroup="sys{}".format(sys_index + 1),
            )
        )

        # Find where to put the arrow
        imag_min_index = np.argmin(imag)
        imag_max_index = np.argmax(imag)
        if np.absolute(imag[imag_max_index]) - np.absolute(imag[imag_min_index]) < 0:
            index = imag_min_index
        else:
            index = imag_max_index

        y_arrow = imag[index]  # Imaginary part
        x_arrow = real[
            index
        ]  # Real part corresponding to the minimal/maximal imaginary part

        # Find the direction of the arrow
        if real[index] - real[index - 1] < 0:
            symbol = "triangle-left"
        else:
            symbol = "triangle-right"

        # Put the arrow head
        fig.add_trace(
            go.Scatter(
                x=[x_arrow],
                y=[y_arrow],
                mode="markers",
                marker=dict(
                    # color=fig.data[sys_index].line.color,
                    color=color_list[sys_index],
                    size=10,
                    symbol=symbol,
                    line_width=2,
                    # line_color=fig.data[sys_index].line.color,
                    line_color=color_list[sys_index],
                ),
                hoverinfo="none",
                showlegend=False,
                legendgroup="sys{}".format(sys_index + 1),
            )
        )

        # Refresh (if needed) the max limit point to have the best zoom for the figure
        if max_limit < np.absolute(imag[index]):
            max_limit = np.absolute(imag[index])
        if max_limit < np.absolute(np.amax(real)):
            max_limit = np.absolute(np.amax(real))
        if max_limit < np.absolute(np.amin(real)):
            max_limit = np.absolute(np.amin(real))

    # Mark the critical point (-1, 0)
    fig.add_trace(
        go.Scatter(
            x=[-1],
            y=[0],
            mode="markers",
            marker=dict(
                color="red",
                size=10,
                symbol="cross-thin",
                line_width=2,
                line_color="red",
            ),
            name="Critical point",
            hoverinfo="none",
            showlegend=False,
        )
    )

    # Formatting of the figure
    fig.update_xaxes(title_text="Real", range=[-max_limit, max_limit])
    fig.update_yaxes(title_text="Imaginary", range=[-max_limit, max_limit])

    # Show the figure if the function is called at the and of the notebook cell
    fig

    return fig


# %% Function to generate a second order transfer function based on its typical
# characteristics.


def generateTfFromCharac(G, wn, zeta):
    """
    Generate a second order transfer function based on its typical
    characteristics.

    Parameters
    ----------
    G: float
        Gain of the transfer function.

    wn: float
        Frequency of the transfer function.

    zeta: float
        Damping coefficient of the transfer function.

    Returns
    -------
    ft: TransferFunction
        The linear system with those charcteristics
    """

    ft = ml.tf([G], [1 / wn**2, 2 * zeta / wn, 1])
    return ft


# %% Function to add noise to a given signal


def addNoise(t, signal, variance=0.05, rndNb=None):
    """
    Add noise to a given signal.

    Parameters
    ----------
    t: 1D array
        Time vector.
    signal: 1D array
        Signal at which to add noise.
    variance: float, optional
        Variance of random numbers. The default is 0.05.
    rndNb: int, optional
        Seed for RandomState. The default is None.

    Returns
    -------
    signal_noisy: 1D array
        Noisy signal.
    """

    if rndNb is not None:
        np.random.seed(rndNb)  # To master the random numbers
    noise = np.random.normal(0, variance, len(signal))
    signal_noisy = signal + noise

    plt.figure()
    plt.plot(t, signal, label="Original")
    plt.plot(t, signal_noisy, label="With noise")

    return signal_noisy


# %% Save data into a csv file


def saveFT(t, y, x=None, name="data"):
    """
    Save the data of the transfert function into a csv file.

    Parameters
    ----------
    t: 1D array
        Time vector.

    y: 1D array
        Response of the system.

    x: 1D array, optional
        Input of the system (default = [0, 1, ..., 1])

    name: String
        Name of the csv file (default = "data").

    Returns
    -------
    None
    """

    if x is None:
        x = np.ones(len(t))
        x[0] = 0
    np.savetxt(
        name + ".csv",
        np.transpose([t, x, y]),
        delimiter=",",
        fmt="%s",
        header="Temps(s),Consigne,Mesure",
        comments="",
    )


# %% Load data from a csv file


def loadFT(file="data.csv"):
    """
    Load the data of the transfert function from a given csv file.

    Parameters
    ----------
    file: String
        Name of the csv file (default = "data.csv").

    Returns
    -------
    None
    """

    # Reading of the data headers with a comma as delimiter
    head = np.loadtxt(file, delimiter=",", max_rows=1, dtype=np.str)
    # Reading of the data
    data = np.loadtxt(file, delimiter=",", skiprows=1, dtype=np.str)

    # Printing of the headers
    print(head)

    # Data selections based on header and convert to float
    # The sign - adapts the input data to be positive
    t = np.asarray(data[:, 0], dtype=np.float, order="C").flatten()
    x = np.asarray(data[:, 1], dtype=np.float, order="C").flatten()
    y = np.asarray(data[:, 2], dtype=np.float, order="C").flatten()

    return [t, x, y]


# %% Function to get the class of a given system.


def getClass(sys):
    """
    Get the class of the given system.

    Parameters
    ----------
    sys: LTI
        System analysed.

    Returns
    -------
    sysClass: int
        Class of the given system.
    """

    __, den = ml.tfdata(sys)
    den = den[0][0]  # To have the array as it's a list with one array
    # Reverse the direction of loop because the smallest power is at the last
    # index
    for sysClass, item in enumerate(reversed(den)):
        if item != 0:
            return sysClass


# %% Function to get the order of a given system.


def getOrder(sys):
    """
    Get the order of the given system.

    Parameters
    ----------
    sys: LTI
        System analysed.

    Returns
    -------
    sysClass: int
        Order of the given system.
    """

    __, den = ml.tfdata(sys)
    den = den[0][0]  # To have the array as it's a list with one array
    sysOrder = len(den) - 1
    return sysOrder


# %% PID Tuner to see the modifications of the PID parameters on a given system.
def pidTuner(H, Kp=1, Ki=0, Kd=0):
    """
    PID Tuner to see the modifications of the PID parameters on a given system.

    Parameters
    ----------
    H: LTI
        Transfert function of the system (open loop) to regulate.

    Kp: float, optionnal
        Proportionnal parameter of the PID controller (default = 1).

    Ki: float, optionnal
        Integral parameter of the PID controller (default = 0).

        Reminder: Ki = Kp/tI

    Kd: float, optionnal
        Derivative parameter of the PID controller (default = 0).

        Reminder: Kd = Kp*tD

    Returns
    -------
    None
    """

    from matplotlib.widgets import Slider, Button, RadioButtons

    # Create the figure
    fig = plt.figure("PID Tuner")
    axGraph = fig.subplots()
    plt.subplots_adjust(bottom=0.3)

    # Frames to contain the sliders
    axcolor = "lightgoldenrodyellow"
    axKp = plt.axes([0.125, 0.2, 0.775, 0.03], facecolor=axcolor)
    axKi = plt.axes([0.125, 0.15, 0.775, 0.03], facecolor=axcolor)
    axKd = plt.axes([0.125, 0.1, 0.775, 0.03], facecolor=axcolor)

    # Slider
    sKp = Slider(axKp, "Kp", Kp / 20, Kp * 20, valinit=Kp)

    if Ki == 0:
        sKi = Slider(axKi, "Ki", 0, 100, valinit=Ki)
    else:
        sKi = Slider(axKi, "Ki", Ki / 20, Ki * 20, valinit=Ki)

    if Kd == 0:
        sKd = Slider(axKd, "Kd", 0, 100, valinit=Kd)
    else:
        sKd = Slider(axKd, "Kd", Kd / 20, Kd * 20, valinit=Kd)

    def update(val):
        KpNew = sKp.val
        KiNew = sKi.val
        KdNew = sKd.val
        c = KpNew * ml.tf(1, 1) + KiNew * ml.tf(1, [1, 0]) + KdNew * ml.tf([1, 0], 1)
        Gbo = c * H
        if radio.value_selected == "Step":
            axGraph.clear()
            plotStep(axGraph, Gbo)
        elif radio.value_selected == "Nichols":
            axGraph.clear()
            plotNichols(axGraph, Gbo)

        fig.canvas.draw_idle()  # Refresh the plots

    sKp.on_changed(update)
    sKi.on_changed(update)
    sKd.on_changed(update)

    # Reset button
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(resetax, "Reset", color=axcolor, hovercolor="0.975")

    def reset(event):
        sKp.reset()
        sKi.reset()
        sKd.reset()

    reset_button.on_clicked(reset)

    def plotNichols(ax, Gbo):
        nichols([Gbo_init, Gbo], NameOfFigure="PID Tuner", ax=ax)
        # Print infos inside the plot
        textInfo = getNicholsTextInfos(Gbo)
        at = AnchoredText(
            textInfo,
            prop=dict(size=10),
            frameon=True,
            loc="lower right",
        )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        axGraph.add_artist(at)

    def plotStep(ax, Gbo):
        Gbf = ml.feedback(Gbo)
        [Y_New, __] = ml.step(Gbf, T)
        ax.plot(
            T, np.linspace(1, 1, len(T)), linestyle=":", lw=1, color="grey"
        )  # 1 line
        ax.plot(T, Y, label="Initial system", lw=1)  # Original
        ax.plot(T, Y_New, label="Modified system", lw=1)
        # l.set_ydata(Y_New)
        ax.set_title("Step Response")
        ax.set_ylabel("Magnitude")
        ax.set_xlabel("Time (seconds)")

    # Print button
    printax = plt.axes([0.6, 0.025, 0.1, 0.04])
    print_button = Button(printax, "Infos", color=axcolor, hovercolor="0.975")

    # Function to create a string with the usefull info fo nichols plot
    def getNicholsTextInfos(Gbo):
        # Extract the gain margin (Gm) and the phase margin (Pm)
        gm, pm, __, __ = ml.margin(Gbo)
        gm = 20 * np.log10(gm)  # Conversion of gm in dB
        return """Phase Margin = {PM}°
Gain Margin = {GM} dB""".format(
            PM=pm, GM=gm
        )

    def printInfos(event):
        KpNew = sKp.val
        KiNew = sKi.val
        KdNew = sKd.val
        print("")  # To let space before the informations
        print("Kp =", KpNew)
        print("Ki =", KiNew)
        print("Kd =", KdNew)
        c = KpNew * ml.tf(1, 1) + KiNew * ml.tf(1, [1, 0]) + KdNew * ml.tf([1, 0], 1)
        print("Corr =", c)
        Gbo = c * H
        print("Gbo =", Gbo)
        if radio.value_selected == "Step":
            Gbf = ml.feedback(Gbo)
            [Y_New, __] = ml.step(Gbf, T)

            # To change the current axes to be the graphs's one
            plt.sca(axGraph)
            stepInfo = step_info(T, Y_New)
            # Printing of the step infos
            printInfo(stepInfo)

        elif radio.value_selected == "Nichols":
            # Extract the gain margin (Gm) and the phase margin (Pm)
            gm, pm, __, __ = ml.margin(Gbo)
            print("Phase Margin =", pm, "°")
            gm = 20 * np.log10(gm)  # Conversion of gm in dB
            print("Gain Margin =", gm, "dB")
            # Plotting
            if pm != math.inf:
                axGraph.plot([-180, -180 + pm], [0, 0], "k-", linewidth=1)
                axGraph.plot(-180 + pm, 0, "ko")
            if gm != math.inf:
                axGraph.plot([-180, -180], [-gm, 0], "k-", linewidth=1)
                axGraph.plot(-180, -gm, "ko")

    print_button.on_clicked(printInfos)

    # Radio button
    rax = plt.axes([0.905, 0.5, 0.09, 0.1], facecolor=axcolor)
    radio = RadioButtons(rax, ("Step", "Nichols"), active=0)

    def changeGraph(label):
        # Get the new parameters values
        KpNew = sKp.val
        KiNew = sKi.val
        KdNew = sKd.val
        c = KpNew * ml.tf(1, 1) + KiNew * ml.tf(1, [1, 0]) + KdNew * ml.tf([1, 0], 1)
        Gbo = c * H
        # Deleting of the graphs
        axGraph.clear()
        # Base for the original graph
        if label == "Step":
            plotStep(axGraph, Gbo)
        elif label == "Nichols":
            plotNichols(axGraph, Gbo)

        fig.canvas.draw_idle()  # To refresh the plots

    radio.on_clicked(changeGraph)

    # Declaration of the transfer function of the system in BO and BF with the
    # given control parameters
    c = Kp * ml.tf(1, 1) + Ki * ml.tf(1, [1, 0]) + Kd * ml.tf([1, 0], 1)
    Gbo_init = c * H
    Gbf_init = ml.feedback(Gbo_init)
    [Y, T] = ml.step(Gbf_init)

    # Plot the step
    plotStep(axGraph, Gbo_init)

    plt.show()

    # It's needed to return those variables to keep the widgets references or
    # they don't work.
    return sKp, print_button, reset_button, radio


# %% Object to store all the characteristics determined.
class TfCharact:

    """
    Object to store all the characteristics determined.

    Attributes
    ----------

    G: float
        Time it takes for the response to rise from 10% to 90% of the
        steady-state response.

    wn: float
        Time it takes for the error e(t) = \|y(t) – yfinal\| between the
        response y(t) and the steady-state response yfinal to fall below 5% of
        the peak value of e(t).

    zeta: float
        Minimum value of y(t) once the response has risen.
    """

    G = None
    wn = None
    zeta = None


# %% Tool to identify a system
def fineIdentification(file, G, wn, zeta, tfCharach):
    """
    Tool to identify a system from its measured step data output.

    Parameters
    ----------
    file: String
        Name of the csv file (default = "data.csv").

    G: float
        Gain of the transfer function.

    wn: float
        Frequency of the transfer function.

    zeta: float
        Damping coefficient of the transfer function.

    tfCharach: TfCharact
        Object to store the characteristics determined.

    Returns
    -------
    None
    """

    from scipy.signal import lti, lsim
    from matplotlib.widgets import Slider, Button

    # Load the data from the given csv file
    t, u, y = loadFT(file)

    # Déclaration de la fonction de transfert
    ft = lti([G], [1 / wn**2, 2 * zeta / wn, 1])
    t_ft, y_ft, _ = lsim(ft, U=u, T=t)  # Réponse à la consigne

    fig, __ = plt.subplots()
    plt.subplots_adjust(bottom=0.3)
    (l,) = plt.plot(t, y, label="Mesures")
    (l,) = plt.plot(t_ft, y_ft, lw=1, color="red")

    axcolor = "lightgoldenrodyellow"
    axamp = plt.axes([0.125, 0.2, 0.775, 0.03], facecolor=axcolor)
    axfreq = plt.axes([0.125, 0.15, 0.775, 0.03], facecolor=axcolor)
    axdamp = plt.axes([0.125, 0.1, 0.775, 0.03], facecolor=axcolor)

    samp = Slider(axamp, "G", G / 10, G * 10.0, valinit=G)
    sfreq = Slider(axfreq, "Wn", 0, wn * 2, valinit=wn)
    sdamp = Slider(axdamp, "Dzeta", zeta / 10, zeta * 10.0, valinit=zeta)

    def update(val):
        amp = samp.val
        omega2 = sfreq.val
        m2 = sdamp.val
        ft = lti([amp], [1 / omega2**2, 2 * m2 / omega2, 1])
        __, y_ft, _ = lsim(ft, U=u, T=t_ft)  # Réponse à la consigne
        l.set_ydata(y_ft)
        fig.canvas.draw_idle()

    samp.on_changed(update)
    sfreq.on_changed(update)
    sdamp.on_changed(update)

    # Reset button
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(resetax, "Reset", color=axcolor, hovercolor="0.975")

    def reset(event):
        samp.reset()
        sfreq.reset()
        sdamp.reset()

    reset_button.on_clicked(reset)

    # Print button
    saveax = plt.axes([0.6, 0.025, 0.1, 0.04])
    save_button = Button(saveax, "Save", color=axcolor, hovercolor="0.975")

    def save(event):
        tfCharach.G = samp.val
        tfCharach.wn = sfreq.val
        tfCharach.zeta = sdamp.val
        print("\nCharacterictics saved!")  # To let space before the informations
        print("G =", tfCharach.G)
        print("wn =", tfCharach.wn)
        print("zeta =", tfCharach.zeta)

    save_button.on_clicked(save)

    plt.show()

    # It's needed to return those variables to keep the widgets references or
    # they don't work.
    return samp, save_button, reset_button


# %% Function to find zeta from the Overshoot given in %
def zetaFromOvershoot(overshoot):
    """
    Function to find zeta from a given relative overshoot.

    Parameters
    ----------
    overshoot: float
        overshoot in % for which we want to find the zeta.

    Returns
    -------
    zeta: float
        zeta corresponding to the given relative overshoot.
    """
    zeta = math.sqrt(
        math.log(overshoot / 100) ** 2 / (math.pi**2 + math.log(overshoot / 100) ** 2)
    )
    return zeta
