import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D


# Plot the acquisition function with the search space bound box and the
# bounding boxes around each observation
def plot_acq(x1g, x2g, acq_upper, X_init, b_init_lower, kernel_k2,
             scale_l, box_len, my_cmap, bounds_all, fig, n_iter, max_iter):

#    fig = plt.figure(figsize=(8, 5))
    if (max_iter % 2) == 1:
        raise AssertionError("max_iter needs to be an even integer!")

    acq2d = fig.add_subplot(2, np.floor(max_iter/2), n_iter)
    myrectangle = patches.Rectangle(b_init_lower, 1.8, box_len, alpha=1,
                                    fill=False, facecolor="#eeefff",
                                    linewidth=3)
    acq2d.add_patch(myrectangle)

    # Plot the contour of the acquisition function
    CS_acq = acq2d.contourf(x1g, x2g, acq_upper.reshape(x1g.shape), 20,
                            cmap=my_cmap, origin='lower')
    plt.contour(CS_acq, levels=CS_acq.levels[::2], colors='r',
                origin='lower', hold='on', alpha=0.5)

    # Plot search space bounding boxes
    X_init_array = np.asarray(X_init)
    acq2d.scatter(X_init_array[:, 0], X_init_array[:, 1], marker='*',
                  s=300, color='red')
    if (n_iter >= 1):
        acq2d.scatter(X_init_array[-1, 0], X_init_array[-1, 1], marker='*',
                      s=300, color='cyan')

    # Plot the bounding boxes around each observation
#    for i in range(len(X_init)):
#        rectangle_temp = patches.Rectangle(X_init[i] - scale_l*kernel_k2,
#                                           2*scale_l*kernel_k2,
#                                           2*scale_l*kernel_k2,
#                                           alpha=0.3, fill=False,
#                                           facecolor="#00ffff", linewidth=3)
#        acq2d.add_patch(rectangle_temp)

    # Plot the overall bounding box
    bounds_ori_all = []
    for n_obs in range(len(X_init)):

        X0 = X_init[n_obs]

        # Set the rectangle bounds
        bounds_ori_temp = np.asarray((X0 - scale_l*kernel_k2,
                                      X0 + scale_l*kernel_k2))
        bounds_ori = bounds_ori_temp.T
        bounds_ori_all.append(bounds_ori)

    temp = np.asarray(bounds_ori_all)
    temp_min = np.min(temp[:, :, 0], axis=0)
    temp_max = np.max(temp[:, :, 1], axis=0)
    bounds_ori = np.stack((temp_min, temp_max)).T

    bounds_temp = np.stack((np.maximum(bounds_ori[:, 0], bounds_all[:, 0]),
                            np.minimum(bounds_ori[:, 1], bounds_all[:, 1])))
    bounds_temp = bounds_temp.T
    rectangle_all = patches.Rectangle(bounds_temp[:, 0],
                                      bounds_temp[0, 1]-bounds_temp[0, 0],
                                      bounds_temp[1, 1]-bounds_temp[1, 0],
                                      alpha=1, fill=False,
                                      edgecolor="#bfbf00", linewidth=3)
    acq2d.add_patch(rectangle_all)

    title_str = 'Iteration {}'.format(n_iter)
    acq2d.set_title(title_str, fontsize=19)
#    acq2d.set_xlim(-5, 10)
#    acq2d.set_ylim(0, 15)
    fig.colorbar(CS_acq, ax=acq2d, shrink=0.9)

    return 0


# Plot the function needs to be evaluated
def plot_original_function(myfunction, bounds, my_cmap):

    func = myfunction.func

    if myfunction.input_dim == 1:
        x = np.linspace(myfunction.bounds['x'][0],
                        myfunction.bounds['x'][1],
                        1000)
        y = func(x)

        fig = plt.figure(figsize=(8, 5))
        plt.plot(x, y)
        strTitle = "{:s}".format(myfunction.name)
        plt.title(strTitle)

    if myfunction.input_dim == 2:

        x1 = np.linspace(bounds[0][0], bounds[0][1], 100)
        x2 = np.linspace(bounds[1][0], bounds[1][1], 100)
        x1g, x2g = np.meshgrid(x1, x2)
        X_plot = np.c_[x1g.flatten(), x2g.flatten()]
        Y_original = func(X_plot)
        Y = ((Y_original-np.mean(Y_original)) /
             (np.max(Y_original)-np.min(Y_original)))

        fig = plt.figure(figsize=(6, 3.5))
        ax3d = fig.add_subplot(1, 1, 1, projection='3d')
        ax3d.plot_surface(x1g, x2g, Y.reshape(x1g.shape), cmap=my_cmap)

        strTitle = "{:s}".format(myfunction.name)
        ax3d.set_title(strTitle)

    return 0


