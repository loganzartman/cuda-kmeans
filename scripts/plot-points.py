import sys
import csv_mt

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

def main():
        tables = csv_mt.read(sys.stdin, parse_float=True)
        assert("points" in tables)
        assert("clusters" in tables)
        print("OK")

        points = tables["points"]
        x = points["dim_0"]
        y = points["dim_1"]

        centroids = tables["clusters"]
        cx = centroids["dim_0"]
        cy = centroids["dim_1"]

        plt.plot(x, y, "x", color="blue")
        plt.plot(cx, cy, "+", color="red")
        plt.savefig("points.png", bbox_inches="tight", dpi=300)
        print("wrote file")

main()
