import matplotlib.pyplot as plt
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

colors = {-1: 'yellow', 0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'brown', 5: 'orange', 6: 'beige', 7: 'magenta', 8: 'gray', 9: 'pink', 10: 'indigo', 11: 'maroon', 12: 'coral', 13: 'lime', 14: 'deepskyblue', 15: 'gold', 16: 'lightgray', 17: 'plum', 18: 'tan', 19: 'yellowgreen', 20: 'darkgreen', 21: 'darkblue', 22: 'lightgreen', 23: 'lightblue', 24: 'fuchsia', 25: 'wheat', 26: 'lawngreen', 27: 'steelblue', 28: 'orchid', 29: 'slategray', 30: 'peru', 31: 'tomato', 32: 'seagreen', 33: 'mediumblue', 34: 'silver', 35: 'firebrick', 36: 'springgreen', 37: 'darkolivegreen', 38: 'dodgerblue', 39: 'crimson', 40: 'indigo', 41: 'orangered', 42: 'aquamarine', 43: 'lemonchiffon', 44: 'gainsboro', 45: 'darkcyan', 46: 'darkmagenta', 47: 'lightpink', 48: 'navajowhite', 49: 'cornflowerblue', 50: 'mistyrose'}


def convert_cylindrical_to_cartesian(r, theta, z):
    '''
    Function for coordinate conversion.
    '''
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = z
    return x, y, z


def visualize_event(data):
    '''
    Function for the plotting of hits in a 3D point cloud.
    
    Code by Yue Zhao.
    '''
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    data['rho'] = np.sqrt(data['x']**2 + data['y']**2+data['z']**2)
    for _, group in data[data['event_id'] == 0].groupby('particle_id'):
        # Sort the group by hit 'rho' values
        group = group.sort_values(by='rho') 

        # Plotting each track
        ax.plot(group['x'], group['y'], group['z'], alpha=0.3, marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def plot_heatmap(parameters, param, name):
    '''
    Function for the generation of a heatmap presenting the predicted versus
    the true track parameters.
    '''
    
    plt.figure()
    param_names = {"radial coefficient":0, "pitch coefficient":1, "azimuthal coefficient":2, "theta":0, "sinphi":1, "cosphi":2, "q":3}
    all_pred, all_true = [], []
    for event_parameters in parameters:
        predicted, true = event_parameters[1][0][:, param_names[param]], event_parameters[2][0][:, param_names[param]]
        all_pred.append(predicted.cpu().tolist())
        all_true.append(true.cpu().tolist())
    all_pred_flattened = [item for sublist in all_pred for item in sublist]
    all_true_flattened = [item for sublist in all_true for item in sublist]

    heatmap, xedges, yedges = np.histogram2d(all_pred_flattened, all_true_flattened, bins=100)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    transposed_heatmap = heatmap.T
    # A hack to make the points with no data white - setting the 0s to nans
    transposed_heatmap[transposed_heatmap == 0.0] = np.nan

    plt.clf()
    plt.imshow(transposed_heatmap, extent=extent, origin='lower')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Regressed {param} vs ground truth")
    plt.colorbar()
    plt.savefig(f"{param}_heatmap_{name}.png")