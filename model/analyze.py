import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from os import path
import math
import cv2

mpl.use("pgf")
pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "pgf.preamble": r"\usepackage[utf8x]{inputenc} \usepackage[T1]{fontenc} \usepackage{siunitx}",  # using this preamble
}

mpl.rcParams.update(pgf_with_latex)

SCRIPT_FILE_DIR = path.dirname(path.realpath(__file__))
DATADIR = path.join(SCRIPT_FILE_DIR, "../data")
PEAK_IND = (2, 7)  # 7 not included, 0-indexed
K = 10  # to be changed
R = 2.5 / 100
RError = 0.05 / 100
η = 0.85
mError = 0.005 / 1000
peakError = 0.5 / 1000


# Load the raw data
def readCSV(fname="data.csv"):
    data = pd.read_csv(path.join(DATADIR, fname))
    return data


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def expectedGradient():
    return 9 * (math.pi) ** 2 * R**2 * η**2 / K


def avgDampingRatio(row: pd.Series):
    ratios = np.array([])
    T = row["Time Period"]

    for i in range(PEAK_IND[0], PEAK_IND[1] - 1):
        for j in range(i + 1, PEAK_IND[1]):
            n = j - i
            delta = (1 / n) * math.log((row.iloc[i]) / row.iloc[j])
            zeta = delta / math.sqrt(4 * (math.pi) ** 2 + delta**2)

            ratios = np.append(ratios, zeta)
    return np.average(ratios)


def getTimes(row: pd.Series):
    T = row["Time Period"]
    for i in range(0, PEAK_IND[0] - PEAK_IND[1]):
        j = i + PEAK_IND[0]
        colName = f"Time at Peak {i + 1}"
        t = (0.25 + i) * T
    return pd.Series([(0.25 + i) * T for i in range(PEAK_IND[1] - PEAK_IND[0])])


def addTimes(df: pd.DataFrame):
    df[[f"Time at Peak {i}" for i in range(1, PEAK_IND[1] - PEAK_IND[0] + 1)]] = (
        df.apply(getTimes, axis=1)
    )
    return df


def processData():
    df = readCSV()
    df.reset_index()

    masses = df["Mass"]

    # df = addTimes(df)
    df["Damping Ratio"] = df.apply(
        avgDampingRatio,
        axis=1,
    )
    df = df.sort_values("Mass")
    return df


def main():
    plt.figure()

    # config
    plt.xlabel(r"m (\si{\kilo\g})")
    plt.ylabel(r"$\zeta$ (unitless ratio)")
    plt.margins(x=2, y=0.2)

    # prepare data
    df = processData()
    x = df["Mass"]
    y = df["Damping Ratio"]
    m, b = np.polyfit(x, y, 1)
    endpoint = x.max()
    x_1 = np.linspace(0, endpoint, 2)
    y_1 = m * x_1 + b

    # scatter plot
    plt.scatter(x, y)

    # # plot the line of best fit
    # (bf,) = plt.plot(x_1, y_1, label="Best fit line")

    # plot the expected line
    (th,) = plt.plot(
        [0, endpoint],
        [0, endpoint * expectedGradient()],
        label="Stoke's Law Prediction",
    )

    plt.legend(handles=[bf, th], frameon=False)

    plt.savefig(path.join(SCRIPT_FILE_DIR, "figures", "graph.png"), dpi=300)


if __name__ == "__main__":
    main()
