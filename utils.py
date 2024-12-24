import json
from intvalpy import Interval, Tol, precision
from intvalpy_fix import IntLinIncR2


def load_data(directory: str, side: str):
    values_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    loaded_data = []

    # init array for loaded data
    for i in range(8):
        loaded_data.append([])
        for j in range(1024):
            loaded_data[i].append([(0, 0) for _ in range(100 * len(values_x))])

    # load data from all jsons from some side and some directory
    for offset, value_x in enumerate(values_x):
        # open new json for specific value_x ars
        data = {}
        with open(
            directory + "/" + str(value_x) + "lvl_side_" + side + "_fast_data.json",
            "rt",
        ) as f:
            data = json.load(f)

        # load values for specific value_x
        for i in range(8):
            for j in range(1024):
                for k in range(len(data["sensors"][i][j])):
                    loaded_data[i][j][offset * 100 + k] = (
                        value_x,
                        data["sensors"][i][j][k],
                    )

    return loaded_data


def amount_of_neg(all_data, coord_x, coord_y):
    x, y = zip(*all_data[coord_y][coord_x])
    eps = 1 / 16384

    # first of all, lets build y_ex and y_in
    x_new = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    y_ex_up = [-float("inf")] * 11
    y_ex_down = [float("inf")] * 11
    y_in_up = [-float("inf")] * 11
    y_in_down = [float("inf")] * 11

    for i in range(len(x_new)):
        y_list = list(y[i * 100 : (i + 1) * 100])
        y_list.sort()
        y_in_down[i] = y_list[25] - eps
        y_in_up[i] = y_list[75] + eps
        y_ex_up[i] = min(y_list[75] + 1.5 * (y_list[75] - y_list[25]), y_list[-1])
        y_ex_down[i] = max(y_list[25] - 1.5 * (y_list[75] - y_list[25]), y_list[0])

    X_mat = []
    Y_vec = []
    for i in range(len(x_new)):
        x_el = x_new[i]
        # y_ex_up >= X_mat * b >= y_ex_down
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_in_down[i], y_in_up[i]])

    # now we have matrix X * b = Y, but with some "additional" rows
    # we can walk over all rows and if some of them is less than 0, we can just remove it at all
    X_mat_interval = Interval(X_mat)
    Y_vec_interval = Interval(Y_vec)
    to_remove = []
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(
        X_mat_interval, Y_vec_interval
    )
    # if Tol value is less than 0, we must iterate over all rows and add some changes to Y_vec, so Tol became 0
    for i in range(len(Y_vec)):
        X_mat_small = Interval([X_mat[i]])
        Y_vec_small = Interval([Y_vec[i]])
        value = Tol.value(X_mat_small, Y_vec_small, b_vec)
        if value < 0:
            to_remove.append(i)
    return len(to_remove)
