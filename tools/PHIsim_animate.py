import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from contextlib import chdir
from tqdm import tqdm

from tools.PHIsim_sim_params import PHIsim_SimulationParams
from tools.fmt_utils import fmt_eng

# we print a warning for large videos, and error/abort for unreasonably large videos
# these limits can be overriden if you *really* want.
DEFAULT_SANITY_CHECK_LIMITS = dict(
    # in number of frames
    warn =  2000,
    error = 20000
)

def animate_video_slice(work_folder, output_filename, 
                        sim_params: PHIsim_SimulationParams,
                        time_window: tuple[float, float], 
                        **kwargs):
    """Animate a time slice of the simulation run by PHIsim. 

    This function is a wrapper around animate_video(), calculating the frame_limits and 
    timestamp_info from simulation parameters. Other parameters of animate_video() can
    be controlled via the kwargs.

    For description of parameters, see animate_video()
    """
    frame_delta_t = sim_params.simulation_time_step() * sim_params.video_N
    video_start = sim_params.video_start * sim_params.simulation_time_step()

    animate_video(work_folder, output_filename, 
                  timestamp_info=(video_start, frame_delta_t),
                  frame_range=time_window_to_video_frames(time_window, sim_params),
                  phisimout_name=sim_params.output_filename, 
                  # I'm assuming here that μm will be a reasonable x-scale for most PHIsim simulation
                  segment_length=(sim_params.simulation_segment_length()*1e6, "μm"), 
                  **kwargs)


type FrameRange = tuple[int|None, int|None] 
"""Range of frames. 'None' simply means "start" or "end" depending on position in the tuple."""

def time_window_to_video_frames(time_window: tuple[float, float], sim_params: PHIsim_SimulationParams) -> FrameRange:
    assert sim_params.video_N > 0, "No video files are stored by this simulation"

    frame_delta_t = sim_params.simulation_time_step() * sim_params.video_N
    video_start = sim_params.video_start * sim_params.simulation_time_step()

    # a couple of sanity checks
    assert time_window[0] >= video_start, "Start time is before video start"
    assert time_window[1] >= time_window[0], "End time is before start time"
    assert time_window[1] <= sim_params.simulation_total_time(), "End time is after video end"

    frame_start = int((time_window[0] - video_start) / frame_delta_t)
    frame_end = int((time_window[1] - video_start) / frame_delta_t)

    # fix potential rounding errors if you end at exactly the end of the simulation
    if frame_end >= sim_params.nr_cycles:
        frame_end = None

    return frame_start, frame_end


"""
The function below is based on a script provided by Erwin Bente (e.a.j.m.bente@tue.nl), 
which was packaged with PHIsim (https://sites.google.com/tue.nl/phisim/home),
and originally adapted the code from Jake Vanderplas, see below:
------------------------------
Matplotlib Animation Example

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
http://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
"""
def animate_video(work_folder, output_filename, 
                  timestamp_info : tuple[float, float],
                  # optional arguments:
                  phisimout_name='PHIsimout.txt', 
                  show_progress=True, show_result=True, print_debug=True,
                  frame_range: FrameRange=(None, None),
                  segment_length: tuple[float, str]|None=None,
                  sanity_limits=DEFAULT_SANITY_CHECK_LIMITS,
                  # additional arguments passed to animation.save()
                  **kwargs):
    """Animate the video produced by PHIsim.

    Recommended encoding is mp4, you need ffmpeg installed for it to work.
    (how to install ffmpeg on Windows: https://phoenixnap.com/kb/ffmpeg-windows)

    Parameters
    ----------
    work_folder: str
        The folder where the PHIsim output is stored.
    output_filename: str
        The filename of the output video, including extension (e.g. "animation.mp4")
    timestamp_info: tuple[float, float]
        The (start, step) timing info of the video. Used for noting the timestamp of a 
        frame in the video (in seconds). 

    phisimout_name: str, optional
        The name of the PHIsim output file (default: 'PHIsimout.txt')
        The video files derive their name from this filename, so if you 
        changed it from the default in your simulations, you need to change it here too.
    segment_length: tuple(float, str), optional
        The length and unit of the simulation segments in the video, used to scale x-axis (default: None).
        If None, x-axis scale is simply the segment number.
    show_progress: bool, optional
        Whether to show a progress bar while encoding the animation(default: True)
    show_result: bool, optional
        Whether to show the result (default: True)
    print_debug: bool, optional
        Whether to print debug information (default: True)
    frame_range: tuple, optional
        A tuple containing the start and end frame (default: (None, None)). Either or both may be specified.
    sanity_limits: dict, optional
        A dictionary containing the following keys and values:
        - 'warn' : [int], prints a warning when the number of frames is larger than this value
        - 'error' : [int], aborts the animation when the number of frames is larger than this value
    **kwargs
        Additional arguments passed to animation.save() 
        (for example, fps=30 will set the animation fps)
    """

    # enter work folder to load data
    with chdir(work_folder):
        video_data = VideoFileNames(phisimout_name)

        video_dat_LRp = np.loadtxt(video_data.filename_LRp, unpack=True, ndmin=2)
        # print(np.shape(video_dat_LRp))
        dimens        = np.shape(video_dat_LRp)
        nr_frames     = dimens[1]
        nr_sl_data    = dimens[0]
        
        # load the other data files with RLp en carriers as well
        video_dat_RLp = np.loadtxt(video_data.filename_RLp, unpack=True, ndmin=2)
        video_dat_car = np.loadtxt(video_data.filename_car, unpack=True, ndmin=2)

    max_LRp = video_dat_LRp.max()
    max_RLp = video_dat_RLp.max()
    max_car = video_dat_car.max()

    max_y = max_LRp
    if max_RLp > max_LRp:
        max_y = max_RLp 

    frame_slice, frame_slice_count = __to_slice_and_num_frames(frame_range, nr_frames)  

    if print_debug:
        print('Nr of frames:', frame_slice_count, 'Nr of segments:', nr_sl_data)

    if sanity_limits:
        if err_limit := sanity_limits.get("error", None):
            if frame_slice_count > err_limit:
                print(f'ERROR: video data would have {frame_slice_count} frames, ABORTING encode') 
                return  
        if warn_limit := sanity_limits.get("warn", None):
            if frame_slice_count > warn_limit:
                print(f'WARNING: video data has {frame_slice_count} frames, resulting file will be large and encoding will take a while')

    # copy data to a 3d array and scale all data to 1
    npdata = np.zeros((nr_sl_data, 3, nr_frames))
    npdata[:,0,frame_slice] = video_dat_LRp[:,frame_slice] / max_y
    npdata[:,1,frame_slice] = video_dat_RLp[:,frame_slice] / max_y
    npdata[:,2,frame_slice] = video_dat_car[:,frame_slice] / max_car

    max_y = 1.0
    # max_y = video_dat_car.max()
    if print_debug:
        print('Maximum value in video data: ', max_y)

    #colours of the lines
    plotlays, plotcols = [0,1,2], ["red","blue","green"]

    # First set up the figure, the axis, and the plot element we want to animate
    max_x = nr_sl_data
    if segment_length:
        max_x *= segment_length[0]

    fig   = plt.figure()
    ax    = plt.axes(xlim=(0, max_x), ylim=(0, max_y))

    lines = []
    for index,lay in enumerate(plotlays):
        lobj = ax.plot([],[],lw=2,color=plotcols[index])[0]
        lines.append(lobj)

    # initialization function: plot the background of each frame
    def init():    
        for line in lines:
            line.set_data([], [])
        return lines

    if not segment_length:
        x = np.linspace(1, nr_sl_data, nr_sl_data)
        x_label = "Segments (#)"
    else:
        x = np.linspace(1, nr_sl_data, nr_sl_data) * segment_length[0]
        x_label = f"Length ({segment_length[1]})"

    def animate(i):
        timestamp = timestamp_info[0] + i * timestamp_info[1]
        ax.set_xlabel(x_label + f' [time {fmt_eng(timestamp, digits=4)}s]')

        for lnum,line in enumerate(lines):        
            line.set_data(x, npdata[:,lnum,i])

        return lines
    
    # show a progress bar if needed - this is recommended as encoding can take a 
    # couple of second up to minutes for longer animations
    frames_iter = range(*frame_slice.indices(nr_frames))
    if show_progress:
        # wrap iter with a tqdm to show a progress bar (tqdm is iterable too)
        frames_iter = tqdm(frames_iter, desc="writing frames", colour="red")

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames_iter, 
                                   interval=100, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    # anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    if output_filename[-4:] != '.mp4':
        print(f"WARNING: output_filename '{output_filename}' does not end with .mp4.")

    # save movie to work_folder
    with chdir(work_folder):
        anim.save(output_filename, **kwargs)

    if show_result:
        plt.show()


def __to_slice_and_num_frames(frame_range : FrameRange, nr_frames:int) -> tuple[slice, int]:
    if frame_range is not None:
        start, end = frame_range
        if start is not None:
            assert start >= 0
        if end is not None:
            assert end <= nr_frames
        frame_slice = slice(start, end) # start and/or end are allowed to be None
    else:
        frame_slice = slice(None) # the "any" slice

    start_idx, stop_idx, _ = frame_slice.indices(nr_frames)

    return frame_slice, (stop_idx - start_idx)


class VideoFileNames:
    def __init__(self, phisimout_name:str):
        # prefix is typically "PHIsimout" but could be different if renamed in simulation settings
        PHIsimout_prefix = phisimout_name[:-4] # strip '.txt' the same way PHIsim does

        self.filename_LRp  = f'{PHIsimout_prefix}_vid_LRp.txt'
        self.filename_RLp  = f'{PHIsimout_prefix}_vid_RLp.txt'
        self.filename_car  = f'{PHIsimout_prefix}_vid_carriers.txt'

from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.axes3d import Axes3D

def wireframe_plot(work_folder, 
                   sim_params: PHIsim_SimulationParams,
                   time_window: tuple[float, float]):
    """Somewhat experimental code to plot an animation as a 3D-plot, where data is plotted on the
    XZ-axes and the Y-axis is time. Not *super* happy with the results yet, but it's a start. 
    """
    
    frame_range = time_window_to_video_frames(time_window, sim_params)

    # enter work folder to load data
    with chdir(work_folder):
        video_data = VideoFileNames(sim_params.output_filename)

        video_dat_LRp = np.loadtxt(video_data.filename_LRp, unpack=True, ndmin=2)
        # print(np.shape(video_dat_LRp))
        dimens        = np.shape(video_dat_LRp)
        nr_frames     = dimens[1]
        nr_sl_data    = dimens[0]
        
        # load the other data files with RLp en carriers as well
        video_dat_RLp = np.loadtxt(video_data.filename_RLp, unpack=True, ndmin=2)
        video_dat_car = np.loadtxt(video_data.filename_car, unpack=True, ndmin=2)

    max_LRp = video_dat_LRp.max()
    max_RLp = video_dat_RLp.max()
    max_car = video_dat_car.max()

    max_y = max_LRp
    if max_RLp > max_LRp:
        max_y = max_RLp 

    frame_slice, frame_slice_count = __to_slice_and_num_frames(frame_range, nr_frames)  

    # build 2D x, y, z arrays for plotting
    z_LRp = video_dat_LRp[:,frame_slice] / max_y
    z_RLp = video_dat_RLp[:,frame_slice] / max_y
    z_car = video_dat_car[:,frame_slice] / max_car

    x_indexes = np.arange(nr_sl_data)
    x_data = x_indexes * sim_params.simulation_segment_length()
    x2d = np.array([x_indexes]*frame_slice_count) * sim_params.simulation_segment_length() # length is constant along y-axis

    timestamps = (np.arange(frame_slice_count) + sim_params.video_start) * sim_params.simulation_time_step() * sim_params.video_N
    half_time_step = (timestamps[1] - timestamps[0]) / 2

    y2d = np.repeat(timestamps[:, np.newaxis], nr_sl_data, axis=1) # timestamp is constant along x-axis

    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111, projection='3d')

    ## code modified from Matplotlib docs "Generate polygons to fill under 3D line graph"
    ## https://matplotlib.org/stable/gallery/mplot3d/polys3d.html#sphx-glr-gallery-mplot3d-polys3d-py
    def polygon_under_graph(x, z):
        """
        Construct the vertex list which defines the polygon filling the space under
        the (x, y) line graph. This assumes x is in ascending order.
        """
        return [(x[0], 0.), *zip(x, z), (x[-1], 0.)]
    
    verts_LRp = [polygon_under_graph(x_data, z_LRp[:,i]) for i in range(0, frame_slice_count)]
    verts_RLp = [polygon_under_graph(x_data, z_RLp[:,i]) for i in range(0, frame_slice_count)]
    #reds = LinearSegmentedColormap.from_list("my_reds", ["red", "darkred"])(np.linspace(0, 1, len(verts_LRp)))
    
    #verts_RLp = [polygon_under_graph(x_data, z_RLp[:,i]) for i in range(frame_slice_count)]
    ax.add_collection3d(PolyCollection(verts_LRp, facecolors="r", alpha=.1, closed=True), zs=timestamps, zdir='y')
    ax.add_collection3d(PolyCollection(verts_RLp, facecolors="b", alpha=.1, closed=True), zs=timestamps+half_time_step, zdir='y')
    
    #for i in range(frame_slice_count):
        #ax.plot(x_data, np.full(nr_sl_data,timestamps[i]), z_LRp[:,i], color="red")
        #ax.plot(x_data, np.full(nr_sl_data,timestamps[i])+half_time_step, z_RLp[:,i], color="blue")
        #ax.plot(x_data, np.full(nr_sl_data,timestamps[i]), z_car[:,i], color="green")
    
    # Set the azimuth and elevation angles
    ax.view_init(40, -20)
    #ax.set_proj_type('ortho')

    #ax.plot_wireframe(x2d, y2d, np.transpose(z_LRp), color="reds", rstride=1, cstride=0)
    #ax.plot_wireframe(x2d, y2d, np.transpose(z_RLp), color="b", rstride=1, cstride=0)
    #ax.plot_wireframe(x2d, y2d, np.transpose(z_car), color="g", rstride=1, cstride=0)

    x_scale=1
    y_scale=2
    z_scale=1

    scale=np.diag([x_scale, y_scale, z_scale, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3]=1.0

    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)

    ax.get_proj=short_proj

    ax.set_xlabel("Length (m)")
    ax.set_ylabel("Time (s)")


def progressive_plot_2d(work_folder, 
                        sim_params: PHIsim_SimulationParams,
                        time_window: tuple[float, float],
                        frameskip: int=5):
    """Somewhat experimental code to plot an animation in a 2D-plot, XY-data is plotted
    to relatively thin slices, which are then stacked on top of each other. This way, both time
    and optical power are projected on the Y-axis.  
    """
        
    frame_range = time_window_to_video_frames(time_window, sim_params)

    # enter work folder to load data
    with chdir(work_folder):
        video_data = VideoFileNames(sim_params.output_filename)

        video_dat_LRp = np.loadtxt(video_data.filename_LRp, unpack=True, ndmin=2)
        # print(np.shape(video_dat_LRp))
        dimens        = np.shape(video_dat_LRp)
        nr_frames     = dimens[1]
        nr_sl_data    = dimens[0]
        
        # load the other data files with RLp en carriers as well
        video_dat_RLp = np.loadtxt(video_data.filename_RLp, unpack=True, ndmin=2)
        video_dat_car = np.loadtxt(video_data.filename_car, unpack=True, ndmin=2)

    max_LRp = video_dat_LRp.max()
    max_RLp = video_dat_RLp.max()
    max_car = video_dat_car.max()

    max_y = max_LRp
    if max_RLp > max_LRp:
        max_y = max_RLp 

    frame_slice, frame_slice_count = __to_slice_and_num_frames(frame_range, nr_frames)  

    # build 2D x, y, z arrays for plotting
    z_LRp = video_dat_LRp[:,frame_slice] / max_y
    z_RLp = video_dat_RLp[:,frame_slice] / max_y
    z_car = video_dat_car[:,frame_slice] / max_car
    
    x_data = np.arange(nr_sl_data) * sim_params.simulation_segment_length()
    delta_t = sim_params.simulation_time_step() * sim_params.video_N
    timestamps = (np.arange(frame_slice_count) + sim_params.video_start) * delta_t

    fig, ax = plt.subplots(tight_layout=True)
    ax.invert_yaxis()

    # we project the z-data for curve N on the range [t_N, t_{N+1}] 
    # that way the curves are flattened and stacked on top of each other
    for i in range(0, frame_slice_count, frameskip):
        # Y = t - normalized data/delta_t 
        # (minus because the Y-axis is pointing down)
        ax.plot(x_data, timestamps[i] - z_LRp[:,i]*delta_t*frameskip, color="red")
        ax.plot(x_data, timestamps[i] - z_RLp[:,i]*delta_t*frameskip, color="blue")
        ax.plot(x_data, timestamps[i] - z_car[:,i]*delta_t*frameskip, color="green")