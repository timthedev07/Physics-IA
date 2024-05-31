import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from os import path
import math
from typing import Union
import cv2


mpl.use("pgf")
pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "pgf.preamble": r"\usepackage[utf8x]{inputenc} \usepackage[T1]{fontenc} \usepackage{siunitx}",  # using this preamble
}

mpl.rcParams.update(pgf_with_latex)

SCRIPT_FILE_DIR = path.dirname(path.realpath(__file__))
DATADIR = path.join(SCRIPT_FILE_DIR, "../data/peaks")
TRIAL_NUM = 5
PEAK_IND = (2, 5)  # 7 not included, 0-indexed
GRAPH_START = 9
K = 100
R = 2.5 / 100
η = 0.85
mError = 0.01 / 1000  # now in kg
peakError = 1 / 1000  # in m


# Load the raw data
def readCSV(fname="data.csv"):
    data = pd.read_csv(path.join(DATADIR, fname))
    return data


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def expectedGradient():
    return 9 * (math.pi) ** 2 * R**2 * η**2 / K


# Calculate the average damping ratio for each row in the dataframe
def avgDampingRatio(row: pd.Series, mode: Union["uncertainties", "vals"]):
    if type(row.iloc[-1]) == str:
        return 0
    ratios = np.array([])  # Create an empty array to store the damping ratios
    zetaSqUncertainties = np.array(
        []
    )  # Create an empty array to store the uncertainties of the damping ratios

    # Iterate over the range of peak indices
    for i in range(PEAK_IND[0], PEAK_IND[1] - 1):
        # Iterate over the range of peak indices starting from i+1
        # the purpose of this double loop is to calculate the damping
        # ratio between every possible pair of peaks
        for j in range(i + 1, PEAK_IND[1]):
            n = j - i  # Calculate the number of cycles between i and j
            A_i, A_j = row.iloc[i], row.iloc[j]  # Get the amplitudes of the peaks

            if type(A_i) != float:
                break

            gamma = A_i / A_j  # Calculate the ratio of the amplitudes of the peaks

            delta = (1 / n) * math.log(
                gamma
            )  # Calculate the logarithmic decrement between the peaks

            errAbsGamma = (A_i / A_j) * (
                (peakError / A_i) + (peakError / A_j)
            )  # Calculate the absolute error of gamma

            # Calculate the absolute error of delta
            errAbsDelta = errAbsGamma / gamma

            zeta = delta / math.sqrt(
                4 * (math.pi) ** 2 + delta**2
            )  # Calculate the damping ratio using the logarithmic decrement

            dzeta_ddelta = (
                4 * math.pi**2 / math.pow(delta**2 + 4 * math.pi**2, 1.5)
            )  # Calculate the derivative of zeta wrt delta

            errAbsZeta = (
                dzeta_ddelta * errAbsDelta
            )  # Calculate the absolute error of zeta

            errAbsZetaSq = (
                2 * zeta * errAbsZeta
            )  # Calculate the absolute error of the squared damping ratio

            ratios = np.append(
                ratios, zeta
            )  # Append the damping ratio to the ratios array

            zetaSqUncertainties = np.append(
                zetaSqUncertainties, errAbsZetaSq
            )  # Append the uncertainty of the damping ratio

    return (
        np.average(ratios) if mode == "vals" else np.average(zetaSqUncertainties)
    )  # Return the average damping ratio or the average
    #    uncertainties of the damping ratios based on the mode


def averageData() -> pd.DataFrame:
    data = (
        pd.concat([readCSV(f"Trial {i}-表格 1.csv") for i in range(1, TRIAL_NUM + 1)])
        .groupby(["Bob", "Mass"], as_index=False, sort=False)[
            [f"Peak {i - PEAK_IND[0] + 1}" for i in range(PEAK_IND[0], PEAK_IND[1])]
        ]
        .mean()
    )
    # convert from gram to kg
    data["Mass"] = data["Mass"] / 1000
    # unit conversion
    for i in range(PEAK_IND[0], PEAK_IND[1]):
        # convert from cm to m
        data[f"Peak {i - PEAK_IND[0] + 1}"] = data[f"Peak {i - PEAK_IND[0] + 1}"] / 100
    return data.round(3)


def processData():
    df = averageData()
    df.reset_index()

    df.iloc[:, 1:] = df.iloc[:, 1:].astype(np.float32)

    df["Damping Ratio"] = df.apply(
        lambda x: avgDampingRatio(x, "vals"),
        axis=1,
    )
    df["Zeta Squared Uncertainties"] = df.apply(
        lambda x: avgDampingRatio(x, "uncertainties"),
        axis=1,
    )
    df = df.sort_values("Mass")
    df["Mass Reciprocal"] = df.apply(lambda row: 1 / (row["Mass"]), axis=1)
    df["Mass Reciprocal Uncertainties"] = df.apply(
        lambda row: np.power(row["Mass"], -2) * mError, axis=1
    )
    df["Zeta Squared"] = df.apply(lambda x: x["Damping Ratio"] ** 2, axis=1)
    return df


def setup(plt, endpoint, y):
    # config
    plt.xlabel(r"$m^{-1}$ (\si{\per\kilo\g})")
    plt.ylabel(r"$\zeta^2$ (unitless ratio)")


def fmt(x):
    return (
        np.format_float_scientific(np.float32(x), unique=False, precision=3)
        .replace("e-0", "\\times 10^{-")
        .replace("e-", "\\times 10^{-")
        + "}"
    )


def bestfit(plt, x, y, x_0, y_0, m, b):
    style = {"color": "brown"}
    r = np.corrcoef(x_0, y_0)
    (bf,) = plt.plot(x, y, label=f"Best fit line")
    t1 = plt.text(16.5, y_0[len(y_0) - 1] + 0.0012, f"$r = {r[0][1]:.3f}$", style)
    t2 = plt.text(
        15.5,
        y_0[len(y_0) - 1],
        f"$y = {fmt(m)}x + {b:.3f}$",
        style,
    )
    t1.set_bbox(dict(facecolor="navajowhite", alpha=1, edgecolor="orange"))
    t2.set_bbox(dict(facecolor="navajowhite", alpha=1, edgecolor="orange"))
    return bf


def theoretical(plt, endpoint):
    m = expectedGradient()
    style = {"color": "g"}
    (th,) = plt.plot(
        [GRAPH_START, endpoint],
        [GRAPH_START * m, endpoint * m],
        "g--",
        label="Stoke's Law Prediction",
    )
    t = plt.text(16.5, 0.008, f"$y = {fmt(m)}x$", style)
    t.set_bbox(dict(facecolor="palegreen", alpha=1, edgecolor="limegreen"))
    return th


def drawMaxMin(plt, x, y, errY):
    n = len(x)
    maxM = (y[n - 1] + errY[n - 1] - (y[0] - errY[0])) / (x[n - 1] - x[0])
    minM = (y[n - 1] - errY[n - 1] - (y[0] + errY[0])) / (x[n - 1] - x[0])
    pA_1 = (x[0], y[0] - errY[0])
    pB_1 = (x[0], y[0] + errY[0])

    maxB = pA_1[1] - maxM * pA_1[0]
    minB = pB_1[1] - minM * pB_1[0]

    xWithZero = np.insert(x, 0, GRAPH_START)

    (minl,) = plt.plot(
        xWithZero,
        minM * xWithZero + minB,
        label="Minimum slope",
        color="mediumslateblue",
        linestyle="dotted",
    )
    (maxl,) = plt.plot(
        xWithZero,
        maxM * xWithZero + maxB,
        label="Maximum slope",
        color="blueviolet",
        linestyle="dotted",
    )

    return (minl, maxl)


def scatter(plt, x, y, df):
    plt.scatter(x, y, c="#000000")
    plt.errorbar(
        x,
        y,
        xerr=df["Mass Reciprocal Uncertainties"],
        yerr=df["Zeta Squared Uncertainties"],
        fmt=" ",
        capthick=1,
        capsize=4,
        ecolor="#000000",
    )


def main():
    plt.figure(0)

    # prepare data
    df = processData()

    x = df["Mass Reciprocal"]
    y = df["Zeta Squared"]
    m, b = np.polyfit(
        x,
        y,
        1,
    )
    endpoint = x.max()
    setup(plt, endpoint, y)
    x_1 = np.linspace(GRAPH_START, endpoint, 2)
    y_1 = m * x_1 + b

    # scatter plot
    scatter(plt, x, y, df)

    # plot the line of best fit
    bf = bestfit(plt, x_1, y_1, x, y, m, b)

    # plot the expected line
    th = theoretical(plt, endpoint)

    plt.legend(handles=[bf, th], frameon=False)

    plt.savefig(path.join(SCRIPT_FILE_DIR, "figures", "graph.png"), dpi=300)

    # plot the max and min lines
    plt.figure(1)
    scatter(plt, x, y, df)
    minl, maxl = drawMaxMin(plt, x, y, df["Zeta Squared Uncertainties"])
    th = theoretical(plt, endpoint)
    plt.legend(handles=[th, minl, maxl], frameon=False)

    setup(plt, endpoint, y)
    plt.savefig(path.join(SCRIPT_FILE_DIR, "figures", "graph_max_min.png"), dpi=300)


if __name__ == "__main__":
    main()
