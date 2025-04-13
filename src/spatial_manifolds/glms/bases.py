import numpy as np

"""
    This is adapted from Uri Eden's spline lab.

    resources:
    - https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0258321
    - https://github.com/MehradSm/Modified-Spline-Regression
    - https://github.com/GiocomoLab/spline-lnp-model/tree/master

"""


def spline_1d(y, num_bins, bounds, s=0.5):

    # S matrix
    S = np.array(
        [[-s, 2 - s, s - 2, s], [2 * s, s - 3, 3 - 2 * s, -s], [-s, 0, s, 0], [0, 1, 0, 0]]
    )

    # Construct spline regressors
    cpts = np.linspace(bounds[0], bounds[1], num_bins)
    bin = cpts[1] - cpts[0]
    cpts_all = np.concatenate(([cpts[0] - bin], cpts, [cpts[-1] + bin]))
    cpts_all[1] = cpts_all[1] - 0.1
    X = np.zeros((len(y), len(cpts_all)))
    num_c_pts = len(cpts_all)  # number of control points in total

    # For each timepoint, calculate the corresponding row of the glm input matrix
    for i in range(len(y)):
        # Find the nearest, and next, control point
        nearest_c_pt_index = np.max(np.where(cpts_all < y[i]))
        nearest_c_pt_time = cpts_all[nearest_c_pt_index]
        next_c_pt_time = cpts_all[nearest_c_pt_index + 1]

        # Compute the alpha (u here)
        u = (y[i] - nearest_c_pt_time) / (next_c_pt_time - nearest_c_pt_time)
        p = np.dot([u**3, u**2, u, 1], S)

        # Fill in the X matrix, with the right number of zeros on either side
        X[i, :] = np.concatenate(
            (
                np.zeros(nearest_c_pt_index - 1),
                p,
                np.zeros(num_c_pts - 4 - (nearest_c_pt_index - 1)),
            )
        )

    return X


def spline_2d(x1, x2, num_bins, bounds, s=0.5):

    # S matrix
    S = np.array(
        [[-s, 2 - s, s - 2, s], [2 * s, s - 3, 3 - 2 * s, -s], [-s, 0, s, 0], [0, 1, 0, 0]]
    )

    # Lay down extra control points
    cpts = np.linspace(bounds[0], bounds[1], num_bins)
    bin = cpts[1] - cpts[0]
    cpts_all = np.concatenate(([cpts[0] - bin], cpts, [cpts[-1] + bin]))
    cpts_all[1] = cpts_all[1] - 0.1
    num_c_pts = len(cpts_all)

    X = np.zeros((len(x1), num_c_pts**2))

    # For each timepoint, calculate the corresponding row of the glm input matrix
    for i in range(len(x1)):
        # For the x dimension
        # Find the nearest, and next, control point
        nearest_c_pt_index_1 = np.max(np.where(cpts_all < x1[i]))
        nearest_c_pt_time_1 = cpts_all[nearest_c_pt_index_1]
        next_c_pt_time_1 = cpts_all[nearest_c_pt_index_1 + 1]

        # Compute the alpha (u here)
        u_1 = (x1[i] - nearest_c_pt_time_1) / (next_c_pt_time_1 - nearest_c_pt_time_1)
        p_1 = np.dot([u_1**3, u_1**2, u_1, 1], S)

        # Fill in the X1 matrix, with the right number of zeros on either side
        X1 = np.concatenate(
            (
                np.zeros(nearest_c_pt_index_1 - 1),
                p_1,
                np.zeros(num_c_pts - 4 - (nearest_c_pt_index_1 - 1)),
            )
        )

        # For the y dimension
        # Find the nearest, and next, control point
        nearest_c_pt_index_2 = np.max(np.where(cpts_all < x2[i]))
        nearest_c_pt_time_2 = cpts_all[nearest_c_pt_index_2]
        next_c_pt_time_2 = cpts_all[nearest_c_pt_index_2 + 1]

        # Compute the alpha (u here)
        u_2 = (x2[i] - nearest_c_pt_time_2) / (next_c_pt_time_2 - nearest_c_pt_time_2)
        p_2 = np.dot([u_2**3, u_2**2, u_2, 1], S)

        # Fill in the X2 matrix, with the right number of zeros on either side
        X2 = np.concatenate(
            (
                np.zeros(nearest_c_pt_index_2 - 1),
                p_2,
                np.zeros(num_c_pts - 4 - (nearest_c_pt_index_2 - 1)),
            )
        )

        # Take the outer product
        X12_op = np.outer(X2, X1)
        X12_op = np.flipud(X12_op)

        X[i, :] = X12_op.flatten()

    return X


def spline_1dc(y, num_bins, bounds, s=0.5):

    # S matrix
    S = np.array(
        [[-s, 2 - s, s - 2, s], [2 * s, s - 3, 3 - 2 * s, -s], [-s, 0, s, 0], [0, 1, 0, 0]]
    )

    # Construct control points
    cpts = np.linspace(bounds[0], bounds[1], num_bins)

    # Construct spline regressors
    X = np.zeros((len(y), len(cpts)))

    # For each timepoint, calculate the corresponding row of the glm input matrix
    for i in range(len(y)):
        # Find the nearest, and next, control point
        if y[i] > 0:
            nearest_c_pt_index = np.max(np.where(cpts < y[i]))
            nearest_c_pt_time = cpts[nearest_c_pt_index]
        else:
            nearest_c_pt_index = len(cpts) - 1
            nearest_c_pt_time = cpts[nearest_c_pt_index]

        if y[i] > cpts[-1] or y[i] == 0:
            next_c_pt_time = cpts[0]
        else:
            next_c_pt_time = cpts[nearest_c_pt_index + 1]

        # Compute the alpha (u here)
        u = np.mod(y[i] - nearest_c_pt_time, np.pi) / np.mod(
            next_c_pt_time - nearest_c_pt_time, np.pi
        )
        p = np.dot([u**3, u**2, u, 1], S)

        # Fill in the X matrix, with the right zeros on either side
        temp_x = np.concatenate((p, np.zeros(len(cpts) - 4)))  # [p1 p2 p3 p4 0 ... 0]
        X[i, :] = np.roll(temp_x, nearest_c_pt_index - 1)

    return X
