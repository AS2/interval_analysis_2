from matplotlib import pyplot as plt

from intvalpy import Interval, Tol, precision
from intvalpy_fix import IntLinIncR2

precision.extendedPrecisionQ = True


# using Tol
def regression_type_1(points, eps: float = 1 / 16384):
    x, y = zip(*points)
    # build intervals radiuses based on eps out of given points
    rads = [eps] * len(y)

    # Build matricies of intervals: each row == some measurment
    X_mat = Interval([[[x_el, x_el], [1, 1]] for x_el in x])
    Y_vec = Interval([[y_el, rads[i]] for i, y_el in enumerate(y)], midRadQ=True)

    # find argmax for Tol
    b_vec, tol_val, _, _, _ = Tol.maximize(X_mat, Y_vec)
    updated = 0
    if tol_val < 0:
        # if Tol value is less than 0, we must iterate over all rows and add some changes to Y_vec, so Tol became 0
        for i in range(len(Y_vec)):
            X_mat_small = Interval([[[x[i], x[i]], [1, 1]]])
            Y_vec_small = Interval([[y[i], rads[i]]], midRadQ=True)
            value = Tol.value(X_mat_small, Y_vec_small, b_vec)
            if value < 0:
                rads[i] = abs(y[i] - (x[i] * b_vec[0] + b_vec[1])) + 1e-8
                updated += 1

    Y_vec = Interval([[y_el, rads[i]] for i, y_el in enumerate(y)], midRadQ=True)
    # find argmax for Tol
    b_vec, tol_val, _, _, _ = Tol.maximize(X_mat, Y_vec)

    return b_vec, rads, updated


# using twin arithmetics
def regression_type_2(points):
    x, y = zip(*points)
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
        Y_vec.append([y_ex_down[i], y_ex_up[i]])
        # y_in_up >= X_mat * b >= y_ex_down
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_ex_down[i], y_in_up[i]])
        # y_ex_up >= X_mat * b >= y_in_down
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_in_down[i], y_ex_up[i]])
        # y_in_up >= X_mat * b >= y_in_down
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_in_down[i], y_in_up[i]])

    # now we have matrix X * b = Y, but with some "additional" rows
    # we can walk over all rows and if some of them is less than 0, we can just remove it at all
    X_mat_interval = Interval(X_mat)
    Y_vec_interval = Interval(Y_vec)
    b_vec, tol_val, _, _, _ = Tol.maximize(X_mat_interval, Y_vec_interval)
    to_remove = []
    if tol_val < 0:
        # if Tol value is less than 0, we must iterate over all rows and add some changes to Y_vec, so Tol became 0
        for i in range(len(Y_vec)):
            X_mat_small = Interval([X_mat[i]])
            Y_vec_small = Interval([Y_vec[i]])
            value = Tol.value(X_mat_small, Y_vec_small, b_vec)
            if value < 0:
                to_remove.append(i)

        for i in sorted(to_remove, reverse=True):
            del X_mat[i]
            del Y_vec[i]

    X_mat_interval = Interval(X_mat)
    Y_vec_interval = Interval(Y_vec)
    b_vec, tol_val, _, _, _ = Tol.maximize(X_mat_interval, Y_vec_interval)

    vertices1 = IntLinIncR2(X_mat_interval, Y_vec_interval)
    vertices2 = IntLinIncR2(X_mat_interval, Y_vec_interval, consistency="tol")

    plt.xlabel("b0")
    plt.ylabel("b1")
    b_uni_vertices = []
    for v in vertices1:
        # если пересечение с ортантом не пусто
        if len(v) > 0:
            x, y = v[:, 0], v[:, 1]
            b_uni_vertices += [(x[i], y[i]) for i in range(len(x))]
            plt.fill(
                x, y, linestyle="-", linewidth=1, color="gray", alpha=0.5, label="Uni"
            )
            plt.scatter(x, y, s=0, color="black", alpha=1)

    b_tol_vertices = []
    for v in vertices2:
        # если пересечение с ортантом не пусто
        if len(v) > 0:
            x, y = v[:, 0], v[:, 1]
            b_tol_vertices += [(x[i], y[i]) for i in range(len(x))]
            plt.fill(
                x, y, linestyle="-", linewidth=1, color="blue", alpha=0.3, label="Tol"
            )
            plt.scatter(x, y, s=10, color="black", alpha=1)

    plt.scatter([b_vec[0]], [b_vec[1]], s=10, color="red", alpha=1, label="argmax Tol")
    plt.legend()
    return (
        b_vec,
        (y_in_down, y_in_up),
        (y_ex_down, y_ex_up),
        to_remove,
        b_uni_vertices,
        b_tol_vertices,
    )
