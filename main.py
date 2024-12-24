from utils import load_data, amount_of_neg
from plotting_utils import build_plots

if __name__ == "__main__":
    side_a_1 = load_data("bin/04_10_2024_070_068", "a")

    # val = [0] * 8
    # for i in range(8):
    #     val[i] = [0] * 1024
    # for j in range(1024):
    #     for i in range(8):
    #         val[i][j] = amount_of_neg(side_a_1, j, i)
    #         print(i, j, val[i][j])

    # build_plots(side_a_1[0][53], 0, 53, "bin/res")
    build_plots(side_a_1[4][1019], 4, 1019, "bin/res")
    # build_plots(side_a_1[3][75], 3, 75, "bin/res")
