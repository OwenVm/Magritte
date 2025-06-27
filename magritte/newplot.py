import matplotlib.pyplot    as plt                  # Mpl plotting
import matplotlib                                   # Mpl
import plotly.graph_objects as go                   # Plotly plotting
import magritte.core        as magritte             # Core functionality
import numpy                as np                   # Data structures
import os                                           # Creating directories
import warnings                                     # Hide warnings
warnings.filterwarnings('ignore')                   # especially for yt

from matplotlib.gridspec  import GridSpec           # Plot layout
from plotly.subplots      import make_subplots      # Plotly subplots
from astropy              import constants, units   # Unit conversions
from scipy.interpolate    import griddata           # Grid interpolation
from palettable.cubehelix import cubehelix2_16      # Nice colormap
from tqdm                 import tqdm               # Progress bars
from ipywidgets           import interact           # Interactive plots
from ipywidgets.embed     import embed_minimal_html # Store interactive plots
from magritte.core        import ImageType, ImagePointPosition  # Image type, point position
from math                 import floor, ceil        # Math helper functions
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec     import GridSpec           # Plot layout
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid
import magritte.tools    as tools      # Save fits
from astropy.io             import fits


# Port matplotlib colormaps to plotly
cubehelix2_16_rgb = []
for i in range(0, 255):
    cubehelix2_16_rgb.append(
        matplotlib.colors.colorConverter.to_rgb(
            cubehelix2_16.mpl_colormap(
                matplotlib.colors.Normalize(vmin=0, vmax=255)(i)
            )
        )
    )

def matplotlib_to_plotly(cmap, pl_entries):
    """
    Converts matplotlib cmap to plotly colorscale.
    """
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
    return pl_colorscale

# Plotly compatible colorscale
cubehelix2_16_plotly = matplotlib_to_plotly(cubehelix2_16.mpl_colormap, 255)

# Plotly standard config
modeBarButtonsToRemove = ['select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'zoom3d', 'pan3d', 'resetCameraDefault3d', 'resetCameraLastSave3d', 'hoverClosest3d', 'orbitRotation', 'tableRotation', 'zoomInGeo', 'zoomOutGeo', 'resetGeo', 'hoverClosestGeo', 'sendDataToCloud', 'hoverClosestGl2d', 'hoverClosestPie', 'toggleHover', 'resetViews', 'toggleSpikelines', 'resetViewMapbox']


def image_mpl(
        model,
        image_nr   =  -1,
        zoom       = 1.3,
        npix_x     = 300,
        npix_y     = 300,
        x_unit     = units.au,
        v_unit     = units.km/units.s,
        method     = 'nearest',
        directory = 'images'
    ):
    """
    Create plots of the channel maps of a synthetic observation (image) with matplotlib.

    Parameters
    ----------
    model : object
        Magritte model object.
    image_nr : int
        Number of the synthetic observation to plot. (Use -1 to indicate the last one.)
    zoom : float
        Factor with which to zoom in on the middel of the image.
    npix_x : int
        Number of pixels in the image in the horizontal (x) direction.
    npix_y : int
        Number of pixels in the image in the vertical (y) direction.
    x_unit : astropy.units object
        Unit of length for the horixontal (x) axis.
    y_unit : astropy.units object
        Unit of length for the vertical (y) axis.
    method : str
        Method to interpolate the scattered intensity data onto a regular image grid.
    directory : str
        Name of the directory in which to store the images.

    Returns
    -------
    None
    """
    # Check if there are images
    if (len(model.images) < 1):
        print('No images in model.')
        return

    # Get path of image directory
    im_dir =  os.path.join(os.path.dirname(os.path.abspath(model.parameters.model_name())), str(directory))

    # If no image directory exists yet
    if not os.path.exists(im_dir):
        # Create image directory
        os.makedirs(im_dir)
        print('Created image directory:', im_dir)

    # Extract data of last image
    imx = np.array(model.images[image_nr].ImX)
    imy = np.array(model.images[image_nr].ImY)
    imI = np.array(model.images[image_nr].I)

    # Workaround for model images
    if (model.images[image_nr].imagePointPosition == ImagePointPosition.AllModelPoints):
    # if (False):
        # Filter imaging data originating from boundary points
        bdy_indices = np.array(model.geometry.boundary.boundary2point)
        imx = np.delete(imx, bdy_indices)
        imy = np.delete(imy, bdy_indices)
        imI = np.delete(imI, bdy_indices, axis=0)

    # Extract the number of frequency bins
    nfreqs = model.images[image_nr].nfreqs

    # Set image boundaries
    deltax = (np.max(imx) - np.min(imx))/zoom
    midx = (np.max(imx) + np.min(imx))/2.0
    deltay = (np.max(imy) - np.min(imy))/zoom
    midy = (np.max(imy) + np.min(imy))/2.0

    x_min, x_max = midx - deltax/2.0, midx + deltax/2.0
    y_min, y_max = midy - deltay/2.0, midy + deltay/2.0

    # Create image grid values
    xs = np.linspace(x_min, x_max, npix_x)
    ys = np.linspace(y_min, y_max, npix_y)

    # Extract the spectral / velocity data
    freqs = np.array(model.images[image_nr].freqs)
    f_ij  = np.mean(freqs)
    velos = (f_ij - freqs) / f_ij * constants.c.to(v_unit).value

    # Interpolate the scattered data to an image (regular grid)
    Is = np.zeros((nfreqs))
    zs = np.zeros((nfreqs, npix_x, npix_y))
    for f in range(nfreqs):
        # Nearest neighbor interpolate scattered image data
        zs[f] = griddata(
            (imx, imy),
            imI[:,f],
            (xs[None,:], ys[:,None]),
            method=method,
            fill_value = 0.0 #for non-nearest neighbor interpolation, otherwise the ceil/floor functions will complain
        )
        Is[f] = np.sum(zs[f])
    Is = Is / np.max(Is)

    # Put zero/negative values to the smallest positive value
    zs[zs<=0.0] = np.min(zs[zs>0.0])
    # Put nan values to smallest positive value
    zs[np.isnan(zs)] = np.min(zs[zs>0.0])

    # Get the logarithm of the data (matplotlib has a hard time handling logarithmic data.)
    log_zs     = np.log10(zs)
    log_zs_min = np.min(log_zs)
    log_zs_max = np.max(log_zs)

    lzmin = ceil (log_zs_min)
    lzmax = floor(log_zs_max)

    lz_25 = ceil (log_zs_min + 0.25*(log_zs_max - log_zs_min))
    lz_50 = ceil (log_zs_min + 0.50*(log_zs_max - log_zs_min))
    lz_75 = floor(log_zs_min + 0.75*(log_zs_max - log_zs_min))

    ticks  = [lzmin, lz_25, lz_50, lz_75, lzmax]
    levels = np.linspace(log_zs_min, log_zs_max, 250)

    figs = []
    gs   = GridSpec(1,2, wspace=.1, width_ratios=[2, 1])

    for f in tqdm(range(nfreqs)):
        fig = plt.figure(dpi=300)
        ax1 = fig.add_subplot(gs[0])
        ax  = ax1.contourf(
            xs / (1.0 * x_unit).si.value,
            ys / (1.0 * x_unit).si.value,
            log_zs[f],
            cmap=cubehelix2_16.mpl_colormap,
            levels=levels
        )
        ax0 = inset_axes(
                  ax1,
                  width="100%",
                  height="5%",
                  loc='lower left',
                  bbox_to_anchor=(0, 1.025, 1, 1),
                  bbox_transform=ax1.transAxes,
                  borderpad=0
        )

        cbar = fig.colorbar(ax, cax=ax0, orientation="horizontal")
        ax0.xaxis.set_ticks_position('top')
        ax0.xaxis.set_label_position('top')
        ax0.xaxis.set_ticks         (ticks)
        ax0.xaxis.set_ticklabels    ([f'$10^{{{t}}}$' for t in ticks])

        ax1.set_aspect('equal')
        ax1.set_xlabel(f'image x [{x_unit}]', labelpad = 10)
        ax1.set_ylabel(f'image y [{x_unit}]', labelpad = 10)

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(velos, Is/np.max(Is))
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.axvline(velos[f], c='red')
        ax2.set_xlabel(f'velocity [{v_unit}]', labelpad=10)
        asp = 2*np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
        ax2.set_aspect(asp)

        if   (model.images[image_nr].imageType == ImageType.Intensity):
            ax0.set_xlabel('Intensity [W m$^{-2}$ sr$^{-1}$ Hz$^{-1}$]', labelpad=11)
            ax2.set_ylabel('Relative intensity',                         labelpad=15)
        elif (model.images[image_nr].imageType == ImageType.OpticalDepth):
            ax0.set_xlabel('Optical depth [.]',      labelpad=11)
            ax2.set_ylabel('Relative optical depth', labelpad=15)

        plt.savefig(f"{im_dir}/image_{f:0>3d}.png", bbox_inches='tight')

        figs.append(fig)

        plt.close()

    # Create a widget for plots
    widget = interact(lambda v: figs[v], v=(0, len(figs)-1))

    return widget


def image_plotly(
        model,
        image_nr   =  -1,
        zoom       = 1.3,
        npix_x     = 300,
        npix_y     = 300,
        x_unit     = units.au,
        v_unit     = units.km/units.s,
        method     = 'nearest',
        width      = 620,   # Yields approx square channel map
        height     = 540    # Yields approx square channel map
    ):
    """
    Create plots of the channel maps of a synthetic observation (image) with plotly.

    Parameters
    ----------
    model : object
        Magritte model object.
    image_nr : int
        Number of the synthetic observation to plot. (Use -1 to indicate the last one.)
    zoom : float
        Factor with which to zoom in on the middel of the image.
    npix_x : int
        Number of pixels in the image in the horizontal (x) direction.
    npix_y : int
        Number of pixels in the image in the vertical (y) direction.
    x_unit : astropy.units object
        Unit of length for the horixontal (x) axis.
    y_unit : astropy.units object
        Unit of length for the vertical (y) axis.
    method : str
        Method to interpolate the scattered intensity data onto a regular image grid.
    width : float
        Width of the resulting figure.
    height : float
        Height of the resulting figure.

    Returns
    -------
    None
    """
    # Check if there are images
    if (len(model.images) < 1):
        print('No images in model.')
        return

    # Get path of image directory
    im_dir = os.path.dirname(os.path.abspath(model.parameters.model_name())) + '/images/'

    # If no image directory exists yet
    if not os.path.exists(im_dir):
        # Create image directory
        os.makedirs(im_dir)
        print('Created image directory:', im_dir)

    # Extract data of last image
    imx = np.array(model.images[image_nr].ImX)
    imy = np.array(model.images[image_nr].ImY)
    imI = np.array(model.images[image_nr].I)

    # Workaround for model images
    if (model.images[image_nr].imagePointPosition == ImagePointPosition.AllModelPoints):
        # Filter imaging data originating from boundary points
        bdy_indices = np.array(model.geometry.boundary.boundary2point)
        imx = np.delete(imx, bdy_indices)
        imy = np.delete(imy, bdy_indices)
        imI = np.delete(imI, bdy_indices, axis=0)

    # Extract the number of frequency bins
    nfreqs = model.images[image_nr].nfreqs

    # Set image boundaries
    deltax = (np.max(imx) - np.min(imx))/zoom
    midx = (np.max(imx) + np.min(imx))/2.0
    deltay = (np.max(imy) - np.min(imy))/zoom
    midy = (np.max(imy) + np.min(imy))/2.0

    x_min, x_max = midx - deltax/2.0, midx + deltax/2.0
    y_min, y_max = midy - deltay/2.0, midy + deltay/2.0

    # Create image grid values
    xs = np.linspace(x_min, x_max, npix_x)
    ys = np.linspace(y_min, y_max, npix_y)

    # Extract the spectral / velocity data
    freqs = np.array(model.images[image_nr].freqs)
    f_ij  = np.mean(freqs)
    velos = (f_ij - freqs) / f_ij * constants.c.to(v_unit).value

    # Interpolate the scattered data to an image (regular grid)
    Is = np.zeros((nfreqs))
    zs = np.zeros((nfreqs, npix_x, npix_y))
    for f in range(nfreqs):
        # Nearest neighbor interpolate scattered image data
        zs[f] = griddata(
            (imx, imy),
            imI[:,f],
            (xs[None,:], ys[:,None]),
            method=method,
            fill_value = 0.0 #for non-nearest neighbor interpolation, otherwise the ceil/floor functions will complain
        )
        Is[f] = np.sum(zs[f])
    Is = Is / np.max(Is)

    # Put zero-values to the smallest non-zero value
    zs[zs<=0.0] = np.min(zs[zs>0.0])
    # Put nan values to smallest positive value
    zs[np.isnan(zs)] = np.min(zs[zs>0.0])

    # Get the logarithm of the data (matplotlib has a hard time handling logarithmic data.)
    log_zs     = np.log10(zs)
    log_zs_min = np.min(log_zs)
    log_zs_max = np.max(log_zs)

    if   (model.images[image_nr].imageType == ImageType.Intensity):
        # Create plotly plot
        fig = make_subplots(
            rows               = 1,
            cols               = 2,
            column_widths      = [0.7, 0.3],
            horizontal_spacing = 0.05,
            subplot_titles     = ['Intensity', ''],
        )
    elif (model.images[image_nr].imageType == ImageType.OpticalDepth):
        # Create plotly plot
        fig = make_subplots(
            rows               = 1,
            cols               = 2,
            column_widths      = [0.7, 0.3],
            horizontal_spacing = 0.05,
            subplot_titles     = ['Optical depth', ''],
        )


    fig.add_vrect(
        row        = 1,
        col        = 1,
        x0         = -1.0e+99,
        x1         = +1.0e+99,
        line_width = 0,
        fillcolor  = "black"
    )

    # Convert to given units
    xs = xs / (1.0 * x_unit).si.value
    ys = ys / (1.0 * x_unit).si.value

    # Convert to given units
    x_max = np.max(xs)
    x_min = np.min(xs)
    y_max = np.max(ys)
    y_min = np.min(ys)

    # Build up plot
    for f in range(nfreqs):
        fig.add_trace(
            go.Heatmap(
                x          = xs       .astype(float),
                y          = ys       .astype(float),
                z          = log_zs[f].astype(float),
                visible    = False,
                hoverinfo  = 'none',
                zmin       = log_zs_min.astype(float),
                zmax       = log_zs_max.astype(float),
                colorscale = cubehelix2_16_plotly,
                showscale  = False
            ),
            row = 1,
            col = 1
        )

        fig.add_trace(
            go.Scatter(
                x          = (velos)        .astype(float),
                y          = (Is/np.max(Is)).astype(float),
                visible    = False,
                hoverinfo  = 'none',
                line_color = '#1f77b4',
                showlegend = False
            ),
            row = 1,
            col = 2
        )

        fig.add_trace(
            go.Scatter(
                x          = np.array([velos[f], velos[f]], dtype=float),
                y          = np.array([-1.0e+10, +1.0e+10], dtype=float),
                visible    = False,
                hoverinfo  = 'none',
                line_color = 'red',
                showlegend = False
            ),
            row = 1,
            col = 2
        )

    # Boxes around plots (more liek mpl)
    fig.update_xaxes(
        showline=True, linewidth=2, linecolor='rgba(1,1,1,1)', mirror=True, ticks='outside'
    )
    fig.update_yaxes(
        showline=True, linewidth=2, linecolor='rgba(1,1,1,1)', mirror=True, ticks='outside'
    )

    v_min = float(min(velos)) * 1.05
    v_max = float(max(velos)) * 1.05

    # Black background for channel map
    fig.add_vrect(
        row        = 1,
        col        = 1,
        x0         = 1000.0*x_min,  # large enough so you don't see edges
        x1         = 1000.0*x_max,  # large enough so you don't see edges
        line_width = 0,
        fillcolor  = "black",
        layer="below"
    )

    # Plot axes
    fig.update_xaxes(
        row        = 1,
        col        = 1,
        title_text = f'image x [{x_unit}]',
        range    = [x_min, x_max],
        showgrid = False,
        zeroline = False
    )
    fig.update_yaxes(
        row         = 1,
        col         = 1,
        title_text  = f'image y [{x_unit}]',
        scaleanchor = "x",
        scaleratio  = 1,
        showgrid    = False,
        zeroline    = False
    )
    fig.update_xaxes(
        row        = 1,
        col        = 2,
        title_text = f'velocity [{v_unit}]',
        range      = [v_min, v_max]
    )

    if   (model.images[image_nr].imageType == ImageType.Intensity):
        fig.update_yaxes(
            row        = 1,
            col        = 2,
            title_text = "Relative intensity",
            side       = 'right',
            range      = [-0.05, +1.05]
        )
    elif (model.images[image_nr].imageType == ImageType.OpticalDepth):
        fig.update_yaxes(
            row        = 1,
            col        = 2,
            title_text = "Relative opacity",
            side       = 'right',
            range      = [-0.05, +1.05]
        )

    # Subplot titles are annotations
    fig.update_annotations(
        font_size = 16,
        borderpad = 7
    )

    fig.update_layout(
        width        = width,
        height       = height,
        plot_bgcolor = 'rgba(0,0,0,0)',
        dragmode     = 'pan',
        font         = dict(family="Calibri", size=14, color='black')
    )

    # Make 3 middle traces visible
    fig.data[3*nfreqs//2  ].visible = True
    fig.data[3*nfreqs//2+1].visible = True
    fig.data[3*nfreqs//2+2].visible = True

    # Create and add slider
    steps = []
    for f in range(nfreqs):
        step = dict(
            method = "restyle",
            args = [
                {"visible": [False] * len(fig.data)},
                {"title": "Channel map: " + str(f)}
            ],
            label = ''
        )
        # Toggle f'th trace to "visible"
        step["args"][0]["visible"][3*f:3*f+3] = [True, True, True]
        steps.append(step)

    sliders = [
        dict(
            active     = nfreqs//2,
            pad        = {"t": 75},
            steps      = steps,
            tickcolor  = 'white',
            transition = {'duration': 0}
        )
    ]

    fig.update_layout(
        sliders=sliders
    )

    # Config for modebar buttons
    config = {
        "modeBarButtonsToRemove": modeBarButtonsToRemove,
        "scrollZoom": True
    }

    # Save figure as html file
    fig.write_html(f"{im_dir}/image.html", config=config)

    return fig.show(config=config)


def plot_velocity_1D(model, xscale='log', yscale='linear'):
    """
    Plot the velocity profile of the model (in 1D, radially).

    Parameters
    ----------
    model : object
        Magritte model object.
    xscale : str
        Scale of the xaxis ("linear", "log", "symlog", "logit", ...)
    yscale : str
        Scale of the yaxis ("linear", "log", "symlog", "logit", ...)

    Returns
    -------
    None
    """
    rs = np.linalg.norm(model.geometry.points.position, axis=1)
    vs = np.linalg.norm(model.geometry.points.velocity, axis=1)

    fig = plt.figure(dpi=300)
    plt.scatter  (rs, vs*constants.c.si.value)
    plt.xlabel('radius [m]',     labelpad=10)
    plt.ylabel('velocity [m/s]', labelpad=10)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.show  ()


def plot_temperature_1D(model, xscale='log', yscale='linear'):
    """
    Plot the temperature profile of the model (in 1D, radially).

    Parameters
    ----------
    model : object
        Magritte model object.
    xscale : str
        Scale of the xaxis ("linear", "log", "symlog", "logit", ...)
    yscale : str
        Scale of the yaxis ("linear", "log", "symlog", "logit", ...)

    Returns
    -------
    None
    """
    rs   = np.linalg.norm(model.geometry.points.position, axis=1)
    temp = np.array      (model.thermodynamics.temperature.gas)

    fig = plt.figure(dpi=300)
    plt.scatter  (rs, temp)
    plt.xlabel('radius [m]',      labelpad=10)
    plt.ylabel('temperature [K]', labelpad=10)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.show  ()


def plot_turbulence_1D(model, xscale='log', yscale='linear'):
    """
    Plot the (micro) turbulence profile of the model (in 1D, radially).

    Parameters
    ----------
    model : object
        Magritte model object.
    xscale : str
        Scale of the xaxis ("linear", "log", "symlog", "logit", ...)
    yscale : str
        Scale of the yaxis ("linear", "log", "symlog", "logit", ...)

    Returns
    -------
    None
    """
    rs     = np.linalg.norm(model.geometry.points.position, axis=1)
    vturb2 = np.array      (model.thermodynamics.turbulence.vturb2)

    fig = plt.figure(dpi=300)
    plt.plot  (rs, np.sqrt(vturb2)*constants.c.si.value)
    plt.xlabel('radius [m]',       labelpad=10)
    plt.ylabel('turbulence [m/s]', labelpad=10)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.show  ()


def plot_number_densities_1D(model, xscale='log', yscale='log'):
    """
    Plot the number densities of all species in the model (in 1D, radially).

    Parameters
    ----------
    model : object
        Magritte model object.
    xscale : str
        Scale of the xaxis ("linear", "log", "symlog", "logit", ...)
    yscale : str
        Scale of the yaxis ("linear", "log", "symlog", "logit", ...)

    Returns
    -------
    None
    """
    rs   = np.linalg.norm(model.geometry.points.position, axis=1)
    abns = np.array      (model.chemistry.species.abundance)
    syms = np.array      (model.chemistry.species.symbol)

    for s in range(1, model.parameters.nspecs()-2):
        fig = plt.figure(dpi=300)
        plt.scatter  (rs, abns[:,s])
        plt.xlabel('radius [m]',                               labelpad=10)
        plt.ylabel(f'{syms[s]} number density [m$^{{{-3}}}$]', labelpad=10)
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.show  ()


def plot_populations_1D(model, lev_min=0, lev_max=7, xscale='log', yscale='log'):
    """
    Plot the relative populations in the model (in 1D, radially).

    Parameters
    ----------
    model : object
        Magritte model object.
    lev_max : int
        Number of levels to plot.
    xscale : str
        Scale of the xaxis ("linear", "log", "symlog", "logit", ...)
    yscale : str
        Scale of the yaxis ("linear", "log", "symlog", "logit", ...)

    Returns
    -------
    None
    """
    rs      = np.linalg.norm(model.geometry.points.position, axis=1)
    npoints = model.parameters.npoints()

    for lspec in model.lines.lineProducingSpecies:
        nlev     = lspec.linedata.nlev
        pops     = np.array(lspec.population).reshape((npoints,nlev))
        pops_tot = np.array(lspec.population_tot)

        plt.figure(dpi=300)

        for i in range(lev_min, min([lev_max, nlev])):
            plt.scatter (rs, pops[:,i]/pops_tot, label=f'i={i}')
            plt.ylabel('fractional level populations [.]', labelpad=10)
            plt.xlabel('radius [m]',                       labelpad=10)
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.legend()
        plt.show()
        
        
########################################################################################################

def select_velocities(velos, s, mm = None):
    '''
    Equally space the velocity bins over the provided minimum and maximum velocities
    n = number of velocity bins
    s = amount of plots to create in the plot (i.e. total amount of plots = s**2)
    mm = minimum and maximum velocities
    '''
    n = len(velos)
    
    if mm == None:
        indexes = np.linspace(0, n-1, s**2)
    else:
        
        mini = -mm
        maxa = mm 
        
        i_s = np.argwhere( np.array(velos) >= mini )[-1][0]
        a_s = np.argwhere( np.array(velos) <= maxa )[0][0]
           
        indexes = np.linspace(i_s, a_s, s**2)   
    
    return np.round(indexes).astype(int)

def save_data(model, name, zoom = 1, res= 512, D = 500):
    tools.save_fits(model, filename=f'{name}.fits', zoom = zoom, npix_x=res, npix_y=res, dpc = D)
    
def get_f_from_v(v, f):
    return f * np.sqrt((1 + v/constants.c.to(units.km/units.s).value) / (1 - v/constants.c.to(units.km/units.s).value))

def get_I(Name):
    '''
    Returns the Flux densities and velocities from a fits file
    '''
    v_unit = units.km/units.s
    
    fits_file = f"{Name}.fits"
    
    hdul = fits.open(fits_file)
    hdr = hdul[0].header
    zs = hdul[0].data

    #velocities and frequencies
    VEL_pixel_size = hdr['CDELT3']/1000
    VEL_center = hdr['CRVAL3']/1000
    VEL_size = hdr['NAXIS3']
    VEL = np.linspace(VEL_center - VEL_pixel_size * (VEL_size - 1) / 2, VEL_center + VEL_pixel_size * (VEL_size - 1) / 2, VEL_size)
    
    rest_f = hdr['RESTFREQ']

    freqs = get_f_from_v(VEL, rest_f)
    nfreqs = len(freqs)   
    
    f_ij  = np.mean(freqs)
    velos = (f_ij - freqs) / f_ij * constants.c.to(v_unit).value

    Is = np.zeros((nfreqs))
    
    for f in range(nfreqs):
        Is[f] = np.sum(zs[f])
    IsW = Is

    #subtract the continuum from the data (should be done better...)
    Ic = (Is[-1] + Is[0])/2

    Is_continuumSubtracted = Is - Ic
    
    #I_normalized = Is_continuumSubtracted / np.amax(Is_continuumSubtracted)
    #Is = I_normalized
    
    return Is_continuumSubtracted, velos 

def double_plot(imx, imy, zoom, size, zs, Is, IsW, velos, cut = 100, name = 'image', trans = (1,0), inclination = 0, v = None, Type = 'au', lim = 14, lx = 2000, ly = 2000):
    '''
    Creates a doule plot, with on the left the channel maps and on the right the spectral line
    '''
    # Set image boundaries
    deltax = (np.max(imx) - np.min(imx))/zoom
    midx = (np.max(imx) + np.min(imx))/2.0
    deltay = (np.max(imy) - np.min(imy))/zoom
    midy = (np.max(imy) + np.min(imy))/2.0

    x_min, x_max = midx - deltax/2.0, midx + deltax/2.0
    y_min, y_max = midy - deltay/2.0, midy + deltay/2.0
    
    zs[zs<=0.0] = np.min(zs[zs>0.0])
    zs[np.isnan(zs)] = np.min(zs[zs>0.0])
    
    # Get the logarithm of the data (matplotlib has a hard time handling logarithmic data.)
    log_zs     = np.log10(zs)
    
    velocities = select_velocities(velos, size, mm = cut)

    gs   = GridSpec(1,2, wspace=.05, width_ratios=[5, 2])

    fig = plt.figure(figsize = (16, 10))
    
    vmin = np.amin(log_zs)
    vmax = np.amax(log_zs)
    
    if v != None:
        vmin, vmax = v
        if vmin == 0:
            vmin = np.amin(log_zs)
        if vmax == 0:
            vmax = np.amax(log_zs)-0.1
            
    vmin -= 0.05
    grid = ImageGrid(fig, gs[0],          # as in plt.subplot(111)
                    nrows_ncols=(size, size),
                    axes_pad=0.05,
                    share_all=True,
                    cbar_location="top",
                    cbar_mode="single",
                    cbar_size="5%",
                    cbar_pad=0.15,
                    
                    )
    
    xmi, xma = x_min, x_max
    ymi, yma = y_min, y_max
    
    for i, ax in enumerate(grid):
        im = ax.imshow(
                log_zs[::-1][velocities[i]],
                cmap='inferno',
                #cmap = 'nipy_spectral',
                extent=[xmi, xma, ymi, yma],
                origin='lower',
                aspect='auto',
                vmin = vmin,
                vmax = vmax,
            )
        
        ax.set_xlim(-lx, lx)
        ax.set_ylim(-ly, ly)
        
        #ax.scatter(0, 0, color = 'yellow', marker = '*', s = 5)
        
        fs = 12
        if size == 1 or size == 2:
            fs = 16
        #only show the ticks in the bottom left corner
        if i < size*(size-1):
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            if Type == 'au':
                ax.set_yticks([-int(lx / 2), 0, int(lx / 2)])
                ax.set_xticks([-int(lx / 2), 0, int(lx / 2)])
            else:
                ax.set_yticks(np.round( np.array([-ly*2/3, 0, ly*2/3]), 2))
                ax.set_xticks(np.round( np.array([-lx*2/3, 0, lx*2/3]), 2))
            
            
        ax.tick_params(axis='both', which='major', labelsize=fs)
            
        #ax.tick_params(axis='both', which='major', labelsize=fs)
        
        ax.text(
                0.05,
                0.95,
                f'{velos[::-1][velocities[i]]:.2f} km/s',
                transform=ax.transAxes,
                fontsize=fs,
                verticalalignment='top',
                color = 'white'
            )
        
        

    cbar = ax.cax.colorbar(im)
    cbar.set_label('$\\log_{10}$(Intensity [Jy/pixel])', fontsize=16)
    ax.cax.toggle_label(True)

    if Type == 'au':
        fig.text(0.39, 0.07, '$x$ [AU]', ha='center', va='center', fontsize=16)
        fig.text(0.13, 0.48, '$y$ [AU]', ha='center', va='center', rotation='vertical', fontsize=16)
    else:
        fig.text(0.39, 0.07, 'Rel. RA  [arcsecond]', ha='center', va='center', fontsize=16)
        fig.text(0.13, 0.48, 'Rel. Dec [arcsecond]', ha='center', va='center', rotation='vertical', fontsize=16)
    
    
    ###########################################################################################################

    ax2 = fig.add_subplot(gs[1])

    lw = 2
    
    ax2.set_title(f'CO $J$={trans[0]}$-${trans[1]}, i = {inclination} $\degree$', fontsize = 16)

    ax2.plot(velos, Is, lw = lw, color = 'black')

    ax2.set_xlabel('Velocity [km/s]', fontsize = 16)
    ax2.set_ylabel('Relative flux density', fontsize = 16)

    ax2.tick_params(axis='both', which='major', labelsize=fs)
    
    ax2.set_xlim(-lim, lim)
    #ax2.invert_xaxis()
    
    ax3 = ax2.twinx()
    ax3.set_ylabel('Flux density [Jy]', rotation = 270, labelpad = 20,fontsize = 16)
    ax3.plot(velos, IsW, lw =lw, color = 'black', alpha = 0)

    plt.tight_layout()

    plt.show()

    fig.savefig(f'{name}.png', dpi = 300, bbox_inches = 'tight')
    
def grid_channel_maps(imx, imy, zs, freqs, nfreqs, size, name, zoom = 1, cut = 0, trans = (1,0), inclination = 0, Type = 'au', lim = 14, lx = 2000, ly = 2000):
    v_unit     = units.km/units.s

    # Extract the spectral / velocity data
    f_ij  = np.mean(freqs)
    velos = (f_ij - freqs) / f_ij * constants.c.to(v_unit).value

    # Interpolate the scattered data to an image (regular grid)
    Is = np.zeros((nfreqs))
    for f in range(nfreqs):
        Is[f] = np.sum(zs[f])
    IsW = Is
    
    Ic = (Is[-1] + Is[0])/2
    I_csubtracted = IsW - Ic
    I_norm = I_csubtracted / np.max(I_csubtracted)
    
    double_plot(imx, imy, zoom, size, zs, I_norm, I_csubtracted, velos, name = name, cut = cut, trans = trans, inclination=inclination, Type=Type, lim = lim, lx = lx, ly = ly)

def make_plot(Name, trans=(1,0), inclination=0, zoom = 1, cut_p = 0, name = 'image', size = 4, casa = False, Type = 'au', lim = 14, lx = 2000, ly = 2000):
    '''
    Creates the double plot for the channel maps and the spectral line
    
    Name = name of the fits file
    trans = transition of the spectral line
    inclination = inclination the model
    zoom = zoom factor for the channel maps
    cut_p = cut off for the channel maps (i.e. inside which velociities to view)
    name = name of the output file
    size = size of the grid for the channel maps
    casa = if the fits file is created with casa (uses a different name for the restfrequency)
    Type = units for the channel maps (au or arcseconds)
    '''
    # Read the FITS file
    fits_file = f"{Name}.fits"
    
    def get_distance(string):
        return int(string.split('_')[-5][-3:])

    hdul = fits.open(fits_file)
    hdr = hdul[0].header
    zs = hdul[0].data
    if casa:
        zs = zs[0]

    #x axis
    RA_pixel_size = hdr['CDELT1']
    RA_center = hdr['CRVAL1']
    RA_size = hdr['NAXIS1']
    RA = np.linspace(RA_center - RA_pixel_size * (RA_size - 1) / 2, RA_center + RA_pixel_size * (RA_size - 1) / 2, RA_size)

    #y axis
    DEC_pixel_size = hdr['CDELT2']
    DEC_center = hdr['CRVAL2']
    DEC_size = hdr['NAXIS2']
    DEC = np.linspace(DEC_center - DEC_pixel_size * (DEC_size - 1) / 2, DEC_center + DEC_pixel_size * (DEC_size - 1) / 2, DEC_size)
    
    #velocities and frequencies
    VEL_pixel_size = hdr['CDELT3']/1000
    VEL_center = hdr['CRVAL3']/1000
    VEL_size = hdr['NAXIS3']
    VEL = np.linspace(VEL_center - VEL_pixel_size * (VEL_size - 1) / 2, VEL_center + VEL_pixel_size * (VEL_size - 1) / 2, VEL_size)

    if casa:
        rest_f = hdr['RESTFRQ']
    else:
        rest_f = hdr['RESTFREQ']

    imx = (((RA*units.degree).to(units.arcsecond) )).to(units.radian)
    imy = (((DEC*units.degree).to(units.arcsecond))).to(units.radian)
    
    distance = get_distance(Name)*units.pc
    
    if len(str(distance)) != 3:
        distance = 500*units.pc
    
    #convert imx imy from arcsecond to au
    if Type == 'au':
        imx = ((np.tan(imx) * distance).to(units.au)).value
        imy = ((np.tan(imy) * distance).to(units.au)).value
    else: 
        imx = (imx).to(units.arcsecond).value
        imy = (imy).to(units.arcsecond).value
    
    freqs = get_f_from_v(VEL, rest_f)
    nfreqs = len(freqs)   
      
    grid_channel_maps(imx, imy, zs, freqs, nfreqs, size, name=name, zoom = zoom, cut = cut_p, trans = trans, inclination = inclination, Type = Type, lim = lim, lx = lx, ly = ly)


def split_string(string):
    return string.split('/')[-1]

def get_inclination(string):
    return string.split('_')[-2]

def get_trans(string):
    if len(string.split('_')[-1]) == 6:
        return string.split('_')[-1][-1]
    else:
        return string.split('_')[-1][-2:]
    
def getNames(data_dir):
    NewNames = []
    for i, string in enumerate(os.listdir(data_dir)):
        new_name = string.split('.')[:-1]
        if new_name[0] == '':
            pass
        else:
            new_el = '.'.join(new_name)
            final_el = os.path.join(data_dir, new_el)
            if final_el not in NewNames:
                NewNames.append(final_el)
    return NewNames

def inclination_to_unitvector(i, phi = 0):
    x, y, z = np.array([np.sin(i), 0, np.cos(i)])
    y = x*np.sin(phi)
    x = x*np.cos(phi)
    return np.array([x, y, z])