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
K = 50  # to be changed
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


def gradientDelta(x, y):
    N = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    S_xx = np.sum((x - x_mean) ** 2)
    S_xy = np.sum((x - x_mean) * (y - y_mean))
    m = S_xy / S_xx
    b = y_mean - m * x_mean

    # Calculate residuals and variance
    residuals = y - (m * x + b)
    sigma_squared = np.sum(residuals**2) / (N - 2)

    # Calculate uncertainty in slope (Delta m)
    Delta_m = np.sqrt(sigma_squared / S_xx)

    # Maximum and minimum slopes
    m_max = m + Delta_m
    m_min = m - Delta_m

    # Corresponding y-intercepts
    b_max = y_mean - m_max * x_mean
    b_min = y_mean - m_min * x_mean
    return ((m_max, b_max), (m_min, b_min))


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

            errAbsZeta = (
                zeta * 2 * (errAbsDelta / delta)
            )  # Calculate the absolute error of zeta

            errAbsZetaSq = (
                2 * (zeta) * (errAbsZeta)
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
    return data


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


def main():
    plt.figure()

    # config
    plt.xlabel(r"$m^{-1}$ (\si{\per\kilo\g})")
    plt.ylabel(r"$\zeta^2$ (unitless ratio)")

    # prepare data
    df = processData()

    x = df["Mass Reciprocal"]
    y = df["Zeta Squared"]
    m, b = np.polyfit(x, y, 1)
    endpoint = x.max()
    x_1 = np.linspace(0, endpoint, 2)
    y_1 = m * x_1 + b

    r = np.corrcoef(x, y)
    (mMax, bMax), (mMin, bMin) = gradientDelta(x, y)
    yMax = mMax * x_1 + bMax
    yMin = mMin * x_1 + bMin

    # scatter plot
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

    # # plot the line of best fit
    (bf,) = plt.plot(x_1, y_1, label="Best fit line")
    (minl,) = plt.plot(x_1, yMin, label="Minimum slope")
    (maxl,) = plt.plot(x_1, yMax, label="Maximum slope")

    # plot the expected line
    (th,) = plt.plot(
        [0, endpoint],
        [0, endpoint * expectedGradient()],
        "--",
        label="Stoke's Law Prediction",
    )

    plt.legend(handles=[bf, th, minl, maxl], frameon=False)

    plt.savefig(path.join(SCRIPT_FILE_DIR, "figures", "graph.png"), dpi=300)


if __name__ == "__main__":
    main()
