from solutions import regression_type_1, regression_type_2
from matplotlib import pyplot as plt
import os
from matplotlib import pyplot as plt
from intvalpy import Interval, Tol, precision

precision.extendedPrecisionQ = True


def build_plots(data, coord_x, coord_y, save_directory: str):
    try:
        os.mkdir(os.path.join(save_directory, f"{coord_x}_{coord_y}"))
    except:
        pass

    # method 1
    b_vec, rads, to_remove = regression_type_1(data)
    x, y = zip(*data)
    plt.figure()
    plt.title("Y(x) method 1 for " + str((coord_x + 1, coord_y + 1)))
    plt.scatter(x, y, label="Intervals mids")
    plt.plot(
        [-0.5, 0.5],
        [b_vec[1] + b_vec[0] * -0.5, b_vec[1] + b_vec[0] * 0.5],
        label="Argmax Tol",
    )
    plt.legend()
    print((coord_x, coord_y), 1, b_vec[0], b_vec[1], to_remove)
    plt.savefig(
        os.path.join(
            save_directory,
            f"{coord_x}_{coord_y}",
            f"{coord_x + 1}_{coord_y + 1}_{1}_res.png",
        )
    )

    plt.figure()
    plt.title("Y(x) - b_0*x - b_1 method 1 for " + str((coord_x + 1, coord_y + 1)))
    for i in range(len(y)):
        plt.plot(
            [i, i],
            [
                y[i] - rads[i] - b_vec[1] - b_vec[0] * x[i],
                y[i] + rads[i] - b_vec[1] - b_vec[0] * x[i],
            ],
            color="k",
            zorder=1,
        )
        plt.plot(
            [i, i],
            [
                y[i] - 1 / 16384 - b_vec[1] - b_vec[0] * x[i],
                y[i] + 1 / 16384 - b_vec[1] - b_vec[0] * x[i],
            ],
            color="blue",
            zorder=2,
        )

    plt.savefig(
        os.path.join(
            save_directory,
            f"{coord_x}_{coord_y}",
            f"{coord_x + 1}_{coord_y + 1}_{1}_ints.png",
        )
    )
    # method 2
    plt.figure()
    plt.title("Uni and Tol method 2 for " + str((coord_x + 1, coord_y + 1)))
    b_vec2, y_in, y_ex, to_remove, b_uni_vertices, b_tol_vertices = regression_type_2(
        data
    )
    plt.legend(loc="upper left")
    print((coord_x, coord_y), 2, b_vec2[0], b_vec2[1], len(to_remove))
    x2 = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    plt.savefig(
        os.path.join(
            save_directory,
            f"{coord_x}_{coord_y}",
            f"{coord_x + 1}_{coord_y + 1}_{2}_tols.png",
        )
    )
    plt.figure()
    plt.title("Y(x) method 2: " + str((coord_x + 1, coord_y + 1)))
    for i in range(len(x2)):
        plt.plot([x2[i], x2[i]], [y_ex[0][i], y_ex[1][i]], color="gray", zorder=1)
        plt.plot([x2[i], x2[i]], [y_in[0][i], y_in[1][i]], color="blue", zorder=2)

    plt.plot(
        [-0.5, 0.5],
        [b_vec2[1] + b_vec2[0] * -0.5, b_vec2[1] + b_vec2[0] * 0.5],
        label="Argmax Tol",
        color="red",
        zorder=1000,
    )

    x2 = [-3] + x2 + [3]

    for i in range(len(x2) - 1):
        x0 = x2[i]
        x1 = x2[i + 1]
        max_idx = 0
        min_idx = 0
        max_val = b_uni_vertices[0][1] + b_uni_vertices[0][0] * (x0 + x1) / 2
        min_val = b_uni_vertices[0][1] + b_uni_vertices[0][0] * (x0 + x1) / 2
        for j in range(len(b_uni_vertices)):
            val = b_uni_vertices[j][1] + b_uni_vertices[j][0] * (x0 + x1) / 2
            if max_val < val:
                max_idx = j
                max_val = val
            if min_val > val:
                min_idx = j
                min_val = val

        y0_low = b_uni_vertices[min_idx][1] + b_uni_vertices[min_idx][0] * x0
        y1_low = b_uni_vertices[min_idx][1] + b_uni_vertices[min_idx][0] * x1
        y0_hi = b_uni_vertices[max_idx][1] + b_uni_vertices[max_idx][0] * x0
        y1_hi = b_uni_vertices[max_idx][1] + b_uni_vertices[max_idx][0] * x1
        plt.fill(
            [x0, x1, x1, x0],
            [y0_low, y1_low, y1_hi, y0_hi],
            facecolor="lightgray",
            linewidth=0,
        )

        max_idx = 0
        min_idx = 0
        max_val = b_tol_vertices[0][1] + b_tol_vertices[0][0] * (x0 + x1) / 2
        min_val = b_tol_vertices[0][1] + b_tol_vertices[0][0] * (x0 + x1) / 2
        for j in range(len(b_tol_vertices)):
            val = b_tol_vertices[j][1] + b_tol_vertices[j][0] * (x0 + x1) / 2
            if max_val < val:
                max_idx = j
                max_val = val
            if min_val > val:
                min_idx = j
                min_val = val

        y0_low = b_tol_vertices[min_idx][1] + b_tol_vertices[min_idx][0] * x0
        y1_low = b_tol_vertices[min_idx][1] + b_tol_vertices[min_idx][0] * x1
        y0_hi = b_tol_vertices[max_idx][1] + b_tol_vertices[max_idx][0] * x0
        y1_hi = b_tol_vertices[max_idx][1] + b_tol_vertices[max_idx][0] * x1
        plt.fill(
            [x0, x1, x1, x0],
            [y0_low, y1_low, y1_hi, y0_hi],
            facecolor="lightblue",
            linewidth=0,
        )

    plt.xlim((-0.6, 0.6))
    plt.ylim((-0.6, 0.6))

    plt.savefig(
        os.path.join(
            save_directory,
            f"{coord_x}_{coord_y}",
            f"{coord_x + 1}_{coord_y + 1}_{2}_res.png",
        )
    )
