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
DATADIR = path.join(SCRIPT_FILE_DIR, "../data/peaks")
TRIAL_NUM = 3
PEAK_IND = (2, 5)  # 7 not included, 0-indexed
K = 10  # to be changed
R = 2.5 / 100
RError = 0.05 / 100
η = 0.85
mError = 0.005  # unit is gram
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

    for i in range(PEAK_IND[0], PEAK_IND[1] - 1):
        for j in range(i + 1, PEAK_IND[1]):
            n = j - i
            delta = (1 / n) * math.log((row.iloc[i]) / row.iloc[j])
            zeta = delta / math.sqrt(4 * (math.pi) ** 2 + delta**2)

            ratios = np.append(ratios, zeta)
    return np.average(ratios)


def averageData():

    data = (
        pd.concat([readCSV(f"Trial {i}-表格 1.csv") for i in range(1, TRIAL_NUM + 1)])
        .groupby(["Bob", "Mass"], as_index=False, sort=False)[
            [f"Peak {i - PEAK_IND[0] + 1}" for i in range(PEAK_IND[0], PEAK_IND[1])]
        ]
        .mean()
    )
    print(data)

    # data = pd.merge(data, tmp, on=["Mass", "Bob"])
    return data


def processData():
    df = averageData()
    print(df)
    df.reset_index()

    df["Damping Ratio"] = df.apply(
        avgDampingRatio,
        axis=1,
    )
    df = df.sort_values("Mass")
    df["Mass Reciprocal"] = df.apply(lambda row: 1 / (row["Mass"] / 1000), axis=1)
    df["Zeta Squared"] = df.apply(lambda x: x["Damping Ratio"] ** 2, axis=1)
    return df


def main():
    plt.figure()

    # config
    plt.xlabel(r"$m^{-1}$ (\si{\per\kilo\g})")
    plt.ylabel(r"$\zeta^2$ (unitless ratio)")

    # prepare data
    df = processData()

    return None

    x = df["Mass Reciprocal"]
    y = df["Zeta Squared"]
    m, b = np.polyfit(x, y, 1)
    endpoint = x.max()
    x_1 = np.linspace(0, endpoint, 2)
    y_1 = m * x_1 + b

    # scatter plot
    plt.scatter(x, y)
    plt.errorbar(x, y, xerr=mError)

    # # plot the line of best fit
    (bf,) = plt.plot(x_1, y_1, label="Best fit line")

    # plot the expected line
    (th,) = plt.plot(
        [0, endpoint],
        [0, endpoint * expectedGradient()],
        "--",
        label="Stoke's Law Prediction",
    )

    plt.legend(handles=[bf, th], frameon=False)

    plt.savefig(path.join(SCRIPT_FILE_DIR, "figures", "graph.png"), dpi=300)


if __name__ == "__main__":
    main()
