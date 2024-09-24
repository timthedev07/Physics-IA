import numpy as np
import math
import pandas as pd
from os import path

SCRIPT_FILE_DIR = path.dirname(path.realpath(__file__))
DATADIR = path.join(SCRIPT_FILE_DIR, "../data/peaks")
r = 0.025  # meter
masses = np.array([52.48, 60.94, 69.72, 86.51, 93.47, 113.27]) / 1000
η = 0.85
A_0 = 0.05  # meter
K = 100

TRIAL_NUM = 5


def theoretical():
    for mass in masses:
        ζ = math.sqrt((9 * (math.pi) ** 2 * r**2 * η**2) / (mass * K))
        ratio = (2 * math.pi * ζ) / (math.sqrt(1 - ζ**2))
        peaks = [A_0 * (ratio) ** i for i in range(1, 4)]
        print(f"Mass: {mass} --- Peaks:", np.array(peaks))


# Load the raw data
def readCSV(fname="data.csv"):
    data = pd.read_csv(path.join(DATADIR, fname))
    return data


def allData() -> pd.DataFrame:
    return [readCSV(f"Trial {i}-表格 1.csv") for i in range(1, TRIAL_NUM + 1)]


def hi():

    for mass in masses:
        ζ = math.sqrt((9 * (math.pi) ** 2 * r**2 * η**2) / (mass * K))
        ratio = math.e ** ((2 * math.pi * ζ) / (math.sqrt(1 - ζ**2)))
        peaks = [A_0 / (ratio) ** i for i in range(1, 4)]
        print(np.array(peaks) * 100)


def main():
    dfs = allData()
    s = ""
    for i in range(dfs[0].shape[0]):
        r = f"{dfs[0].iloc[i, 1]} "
        for j in range(3):
            for k in range(TRIAL_NUM):
                r += f" & {dfs[k].iloc[i, 2 + j]}"
        print(r + " \\\\ \\hline")

    theoretical()


if __name__ == "__main__":
    main()
