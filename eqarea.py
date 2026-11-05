import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import matplotlib.colors as mcolors

def asarray_1d_float(x):
    """
    Verify if x can be converted to a Numpy array 1D containing floats. 
    """
    part_1 = "x could not be converted to a "
    part_2 = "1D "
    part_3 = "numpy array of floats"
    try:
        X = np.atleast_1d(np.asarray(x, dtype=float))
    except (TypeError, ValueError) as e:
        error_message = part_1 + part_3
        raise ValueError(error_message)
    if (X.ndim != 1):
        error_message = part_1 + part_2 + part_3
        raise ValueError(error_message)
    return X


def di2xyz(dec, inc):
    """
    Convert declination and inclination values into 
    their corresponding Cartesian coordinates.

    Parameters
    ----------
    dec, inc : 1D arrays or floats
        The declination and inclination values in degrees.

    Returns
    -------
    x, y, z: 1D arrays
        NumPy arrays 1D with the computed Cartesian coordinates.
    """

    # Convert input to 1D numpy arrays of floats
    D = asarray_1d_float(dec)
    I = asarray_1d_float(inc)
    if D.size != I.size:
        raise ValueError("dec and inc must have the same number of elements.")
    # Convert degrees to radians
    D = np.deg2rad(D)
    I = np.deg2rad(I)
    # Compute the Cartesian coordinates
    x = np.cos(I) * np.cos(D)
    y = np.cos(I) * np.sin(D)
    z = np.sin(I)
    return x, y, z


def xyz2equal_area_polar(x, y, z):
    """
    Compute the coordinates x and y of a polar equal-area projection 
    from their corresponding Cartesian coordinates x, y and z.

    Parameters
    ----------
    x, y, z : 1D arrays
        The Cartesian coordinates.

    Returns
    -------
    x_eqarea, y_eqarea: 1D arrays
        NumPy arrays 1D or floats with the projected coordinates.
    """

    # Verify the Cartesian coordinates
    x = asarray_1d_float(x)
    y = asarray_1d_float(y)
    z = asarray_1d_float(z)
    N = x.size
    if (y.size != N) or (z.size != N):
        raise ValueError("x, y and z must have the same number of elements.")

    # Compute the equal area radius
    r_xy = np.sqrt(1 - np.abs(z))
    h_xy = np.sqrt(x**2 + y**2)
    # Identify where h = 0 to avoid division by zero
    zero_h = (np.abs(h_xy) <= 1e-10)
    # Compute the projected coordinates
    x_eqarea = np.empty_like(x)
    y_eqarea = np.empty_like(x)
    x_eqarea[zero_h] = 0.
    y_eqarea[zero_h] = 0.
    x_eqarea[~zero_h] = ( r_xy[~zero_h] / h_xy[~zero_h] ) * x[~zero_h]
    y_eqarea[~zero_h] = ( r_xy[~zero_h] / h_xy[~zero_h] ) * y[~zero_h]
    return x_eqarea, y_eqarea


def xyz2equal_area_equatorial(x, y, z):
    """
    Compute the coordinates x and y of a equatorial equal-area projection 
    from their corresponding Cartesian coordinates x, y and z.

    Parameters
    ----------
    x, y, z : 1D arrays
        The Cartesian coordinates.

    Returns
    -------
    y_eqarea, z_eqarea: 1D arrays or floats
        NumPy arrays 1D or floats with the projected coordinates.
    """

    # Verify the Cartesian coordinates
    x = asarray_1d_float(x)
    y = asarray_1d_float(y)
    z = asarray_1d_float(z)
    N = x.size
    if (y.size != N) or (z.size != N):
        raise ValueError("x, y and z must have the same number of elements.")

    # Compute the equal area radius
    r_yz = np.sqrt(1 - np.abs(x))
    h_yz = np.sqrt(y**2 + z**2)
    # Identify where h = 0 to avoid division by zero
    zero_h = (np.abs(h_yz) <= 1e-10)
    # Compute the projected coordinates
    y_eqarea = np.empty_like(x)
    z_eqarea = np.empty_like(x)
    y_eqarea[zero_h] = 0.
    z_eqarea[zero_h] = 0.
    y_eqarea[~zero_h] = ( r_yz[~zero_h] / h_yz[~zero_h] ) * y[~zero_h]
    z_eqarea[~zero_h] = ( r_yz[~zero_h] / h_yz[~zero_h] ) * z[~zero_h]
    return y_eqarea, z_eqarea


def plot_net(
    ax, 
    dec_limits = (-180., 180), 
    dec_num_intervals = 36, 
    inc_limits = (0., 90.), 
    inc_num_intervals = 10, 
    projection = ("eqarea", "polar"),
    ):
    """
    Plot the grid lines (net) for declination and inclination values.

    Parameters
    ----------
    ax: Axes
        Individual matplotlib plotting area.
    dec_limits : 2-tuple or 2-list
        Minimum and maximum values of declination
        values in degrees. Default is (-180., 180.).
    dec_num_intervals: integer
        Number of declination intervals from minimum to maximum. Default is 36.
    inc_limits : 2-tuple or 2-list
        Minimum and maximum values of inclination
        values in degrees. Default is (-90., 90.).
    inc_num_intervals: integer
        Number of inclination intervals from minimum to maximum. Default is 18.
    projection: 2-tuple or 2-list
        Defines the projection type. The possible options are:
        ("none", "polar") -> polar net without projection.
        ("none", "equatorial") -> equatorial net without projection.
        ("eqarea", "polar") -> polar equal-area net.
        ("eqarea", "equatorial") -> equatorial equal-area net.
        Default is ("eqarea", "polar").
    """
    if not isinstance(ax, Axes):
        raise ValueError("ax must be an individual matplotlib plotting area.")
    if not isinstance(dec_limits, (tuple, list)):
        raise ValueError("dec_limits must be a tuple or list.")
    if len(dec_limits) != 2:
        raise ValueError("dec_limits must have two elements.")    
    if not all(isinstance(d, (int, float)) for d in dec_limits):
        raise ValueError("dec_limits values must be floats or integers.")
    if dec_limits[1] <= dec_limits[0]:
        raise ValueError("dec_limits[0] must be smaller than dec_limits[1].")
    if (not isinstance(dec_num_intervals, int)) or (dec_num_intervals <= 0):
        raise ValueError("dec_num_intervals must a positive integer.")
    if not isinstance(inc_limits, (tuple, list)):
        raise ValueError("inc_limits must be a tuple or list.")
    if len(inc_limits) != 2:
        raise ValueError("inc_limits must have two elements.")    
    if not all(isinstance(i, (int, float)) for i in inc_limits):
        raise ValueError("inc_limits values must be floats or integers.")
    if inc_limits[1] <= inc_limits[0]:
        raise ValueError("inc_limits[0] must be smaller than inc_limits[1].")
    if (not isinstance(inc_num_intervals, int)) or (inc_num_intervals <= 0):
        raise ValueError("inc_num_intervals must a positive integer.")
    if not isinstance(projection, (tuple, list)):
        raise ValueError("projection must be a tuple or list")
    if len(projection) != 2:
        raise ValueError("projection must have two elements.")
    if any(p not in ["none", "eqarea", "polar", "equatorial"] for p in projection):
        raise ValueError("Invalid projection.")

    # Set axes scale
    ax.axis('scaled')
    
    # Declination and inclination values
    decs = np.linspace(dec_limits[0], dec_limits[1], 100)
    incs = np.linspace(inc_limits[0], inc_limits[1], 50)

    # Constant inclination lines
    for inc in np.linspace(inc_limits[0], inc_limits[1], inc_num_intervals+1):
        # Compute the Cartesian coordinates
        x, y, z = di2xyz(
            decs, np.broadcast_to(inc, (decs.size,))
        )
        if projection[0] == "none":
            if projection[1] == "polar":
                c2, c1 = x, y
            else: # projection[1] == "equatorial"
                c1, c2 = y, z
        else: # projection[0] == "eqarea"
            if projection[1] == "polar":
                c2, c1 = xyz2equal_area_polar(x, y, z)
            else: # projection[1] == "equatorial"
                c1, c2 = xyz2equal_area_equatorial(x, y, z)
        # Plot the line with constant inclination inc
        ax.plot(c1, c2, linestyle='-', linewidth=0.3, color=3*(0.3,))

    # Constant declination lines
    for dec in np.linspace(dec_limits[0], dec_limits[1], dec_num_intervals+1):
        # Compute the Cartesian coordinates
        x, y, z = di2xyz(
            np.broadcast_to(dec, (incs.size,)), incs
        )
        if projection[0] == "none":
            if projection[1] == "polar":
                c2, c1 = x, y
            else: # projection[1] == "equatorial"
                c1, c2 = y, z
        else: # projection[0] == "eqarea"
            if projection[1] == "polar":
                c2, c1 = xyz2equal_area_polar(x, y, z)
            else: # projection[1] == "equatorial"
                c1, c2 = xyz2equal_area_equatorial(x, y, z)
        # Plot the line with constant declination dec
        ax.plot(c1, c2, linestyle='-', linewidth=0.3, color=3*(0.3,))

    # Set axes limits
    if projection[1] == "polar":
        ax.set_xlim(-1.1, 1.1) # original y axis (absissa - horizontal)
        ax.set_ylim(-1.1, 1.1) # original x axis (ordinate - vertical)
    else: # projection[1] == "equatorial"
        ax.set_xlim(-1.1, 1.1) # original y axis (absissa - horizontal)
        ax.set_ylim( 1.1,-1.1) # inverted z axis (ordinate - vertical)

    # Remove axes
    ax.axis("off")


def plot_points(ax, dec, inc, projection = ("eqarea", "polar"), size=7, color='k', label=None):
    """
    Plot points defined by declination and inclination values.

    Parameters
    ----------
    ax: Axes
        Individual matplotlib plotting area.
    dec, inc: NumPy arrays 1D
        NumPy arrays 1D containing the declination and inclination of the points.
    projection: 2-tuple or 2-list
        Defines the projection type. The possible options are:
        ("none", "polar") -> polar net without projection.
        ("none", "equatorial") -> equatorial net without projection.
        ("eqarea", "polar") -> polar equal-area net.
        ("eqarea", "equatorial") -> equatorial equal-area net.
        Default is ("eqarea", "polar").
    size: integer
        Positive integer defining the marker size.
    color: matplotlib color
        Valid matplotlib color that can be verified with matplotlib.colors.mcolors.
        Default is black 'k'.
    label : string
        Define a label for the plotted points. Default is None.
    """
    if not isinstance(ax, Axes):
        raise ValueError("ax must be an individual matplotlib plotting area.")
    # dec and inc are verified within the 'di2xyz' function
    if not isinstance(projection, (tuple, list)):
        raise ValueError("projection must be a tuple or list")
    if len(projection) != 2:
        raise ValueError("projection must have two elements.")
    if any(p not in ["none", "eqarea", "polar", "equatorial"] for p in projection):
        raise ValueError("Invalid projection.")
    if (not isinstance(size, int)) or (size <= 0):
        raise ValueError("size must be a positive integer.")
    if not mcolors.is_color_like(color):
        raise ValueError("Invalid matplotlib color.")
    if (not isinstance(label, str)) and (label is not None):
        raise ValueError("label must be a string or None.")

    # Compute the Cartesian coordinates
    x, y, z = di2xyz(dec, inc)

    if projection[1] == "polar":
        if projection[0] == "none":
            c2, c1 = x, y
        else: # projection[1] == "eqarea"
            c2, c1 = xyz2equal_area_polar(x, y, z)
        # negative z
        negative_z = (z < 0)
        # Plot points with negative z
        if any(negative_z):
            ax.scatter(
                c1[negative_z], c2[negative_z], 
                s=size, marker='o', edgecolor=color, facecolor='none', linewidths=1, label=label
                )
        # Plot points with positive z
        if any(~negative_z):
            ax.scatter(
                c1[~negative_z], c2[~negative_z], 
                s=size, marker='o', edgecolor=color, facecolor=color, linewidths=1, label=label
                )
    else: # projection[1] == "equatorial"
        if projection[0] == "none":
            c1, c2 = y, z
        else: # projection[1] == "eqarea"
            c1, c2 = xyz2equal_area_equatorial(x, y, z)
        # negative x
        negative_x = (x < 0)
        # Plot points with negative x
        if any(negative_x):
            ax.scatter(
                c1[negative_x], c2[negative_x], 
                s=size, marker='o', edgecolor=color, facecolor='none', linewidths=1, label=label
                )
        # Plot points with positive x
        if any(~negative_x):
            ax.scatter(
                c1[~negative_x], c2[~negative_x], 
                s=size, marker='o', edgecolor=color, facecolor=color, linewidths=1, label=label
                )
