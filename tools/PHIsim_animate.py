import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from contextlib import chdir
from tqdm import tqdm
from dataclasses import dataclass

from tools.PHIsim_sim_params import PHIsim_SimulationParams
from tools.PHIsim_dispatcher import PHIsim_Result
from tools.fmt_utils import fmt_eng, SI_prefix_scaling

# we print a warning for large videos, and error/abort for unreasonably large videos
# these limits can be overriden if you *really* want.
DEFAULT_SANITY_CHECK_LIMITS = dict(
    # in number of frames
    warn =  2000,
    error = 20000
)

"""
Some of the code below is based on a script provided by Erwin Bente (e.a.j.m.bente@tue.nl), 
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

def animate_video_with_IO(work_folder, output_filename, 
                          sim_params: PHIsim_SimulationParams,
                          time_window: tuple[float, float], 
                          **kwargs):
    """Animate a time slice of the simulation run by PHIsim. 
    
    Currently, this is designed to animate a 3-panel video of the laser, with
    LR input on the left panel, LR output on the right panel, and the laser in the center.

    If you have a different use-case, and you're not me (is anyone using this code?)
    feel free to experiment with the code or send me a question.

    See the constructor for additional optional arguments passed via kwargs. 
    """
    _VideoAnimator(work_folder, output_filename, sim_params, time_window, **kwargs).animate()


class _VideoAnimator:
    """Helper class to animate a time slice of the simulation run by PHIsim. """	

    @dataclass
    class VideoData:
        LRp : np.ndarray
        RLp : np.ndarray
        car : np.ndarray

    def __init__(self, work_folder, output_filename, 
                 sim_params: PHIsim_SimulationParams,
                 time_window: tuple[float, float], 
                 **kwargs):
        
        self.work_folder = work_folder
        self.output_filename = output_filename
        self.sim_params = sim_params
        self.time_window = time_window

        self.frame_range = time_window_to_video_frames(time_window, sim_params)
        self.frame_delta_t = sim_params.simulation_time_step() * sim_params.video_N
        self.video_start_t = sim_params.video_start * sim_params.simulation_time_step()

        self.laser_pigtail = kwargs.pop("laser_pigtail", None) # in meter

        # I'm assuming that mW, μm and ns will be a reasonable x/y-scale for most PHIsim simulations
        # but this can be overriden via kwargs
        self.segment_unit    = kwargs.pop("segment_unit", "μm")
        self.time_unit       = kwargs.pop("segment_unit", "ns")
        self.power_unit      = kwargs.pop("power_unit",   "mW")
        self.segment_scaling = self._parse_scaling(self.segment_unit, "m") 
        self.time_scaling    = self._parse_scaling(self.time_unit,    "s")
        self.power_scaling   = self._parse_scaling(self.power_unit,   "W")

        # consume optional debug flags from kwargs
        self.show_progress = kwargs.pop("show_progress", True)
        self.show_result   = kwargs.pop("show_result",   True) 
        self.print_debug   = kwargs.pop("print_debug",   True)
        self.sanity_limits = kwargs.pop("sanity_limits", DEFAULT_SANITY_CHECK_LIMITS)

        # rest of kwargs passed to animation.save()
        self.anim_save_kwargs = kwargs


    def _parse_scaling(self, unit, base_unit):
        """Parse unit and return scaling factor to display 'base_unit' quantities in 'unit'.
        For example, for a prefix with scale 1e-3 (mW, mm, ...), we need to multiply everything with 1e3.
        """ 
        if unit == base_unit:
            return 1

        if  len(unit) != len(base_unit) + 1:
            raise ValueError(f"unknown unit {unit} (base {base_unit}), expect unit to be SI-prefix + base_unit")
        
        try:
            return 1/SI_prefix_scaling[unit[0]]  
        except KeyError:
            raise ValueError(f"unknown unit {unit} - {unit[0]} is not a valid SI prefix")


    def load_video_data(self) -> VideoData:
        with chdir(self.work_folder):
            video_names = VideoFileNames(self.sim_params.output_filename)
            video_dat_LRp = np.loadtxt(video_names.filename_LRp, unpack=True, ndmin=2)
            video_dat_RLp = np.loadtxt(video_names.filename_RLp, unpack=True, ndmin=2)
            video_dat_car = np.loadtxt(video_names.filename_car, unpack=True, ndmin=2)
 
        return _VideoAnimator.VideoData(video_dat_LRp, video_dat_RLp, video_dat_car)
    
    def load_input_output_power(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with chdir(self.work_folder):
            P_LR_in, F_LR_in, P_RL_in, F_RL_in = np.loadtxt(self.sim_params.signal_input_file, dtype=float, unpack=True, ndmin=2)
            P_LR_out, F_LR_out, P_RL_out, F_RL_out = np.loadtxt(self.sim_params.output_filename, dtype=float, unpack=True, ndmin=2)

        return P_LR_in, P_LR_out, P_RL_in, P_RL_out

    
    def check_frame_limits(self, frame_count):
        if self.sanity_limits:
            if err_limit := self.sanity_limits.get("error", None):
                if frame_count > err_limit:
                    print(f'ERROR: video data would have {frame_count} frames, ABORTING encode') 
                    return  
            if warn_limit := self.sanity_limits.get("warn", None):
                if frame_count > warn_limit:
                    print(f'WARNING: video data has {frame_count} frames, resulting file will be large and encoding will take a while')


    def show_laser_ends(self, ax: plt.Axes):
        if self.laser_pigtail:
            x_len = self.laser_pigtail * self.segment_scaling
            end_x = ax.get_xlim()[1] - x_len

            ax.axvline(x_len, color="gray", linestyle="--")
            ax.axvline(end_x, color="gray", linestyle="--")

            # update ticks such that the laser starts at 0 and remove labels outside the laser
            ticks_x = np.arange(x_len, end_x, 250) # TODO make tick spacing a parameter?
            
            ax.set_xticks(ticks_x, (ticks_x-x_len).astype(int))


    def make_room_for_legend(self, *axes:plt.Axes):
        # shrink current axis's height by 10% to make some room for the legend at the top
        for ax in axes:
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.9])


    ### main animation function where the magic happens
    def animate(self):
        video_data = self.load_video_data()
        P_LR_in, P_LR_out, P_RL_in, P_RL_out = self.load_input_output_power()

        num_segments, num_frames = np.shape(video_data.LRp)

        max_LRp = video_data.LRp.max()
        max_RLp = video_data.RLp.max()
        max_car = video_data.car.max()

        max_y = max(max_LRp, max_RLp)

        frame_slice, frame_slice_count = _to_slice_and_num_frames(self.frame_range, num_frames)  

        if self.print_debug:
            print('Nr of frames:', frame_slice_count, 'Nr of segments:', num_segments)

        self.check_frame_limits(frame_slice_count)

        # copy data to a 3d array and scale all data to 1
        npdata = np.zeros((num_segments, 3, num_frames))
        npdata[:,0,frame_slice] = video_data.LRp[:,frame_slice] / max_y
        npdata[:,1,frame_slice] = video_data.RLp[:,frame_slice] / max_y
        npdata[:,2,frame_slice] = video_data.car[:,frame_slice] / max_car

        max_y = 1.0
        # max_y = video_data.car.max()
        if self.print_debug:
            print('Maximum value in video data: ', max_y)

        # First set up the figure, the axis, and the plot element we want to animate
        fig, (axin, ax, axout) = plt.subplots(1, 3, figsize=(12, 5), tight_layout=True)

        # the center plot has the laser animation
        max_x = num_segments * self.segment_scaling * self.sim_params.simulation_segment_length() 

        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_y)

        self.show_laser_ends(ax)
        
        # initialize one "empty" line for LR, RL and carrier
        lines = []
        for color in ("blue", "red", "green"):
            lobj = ax.plot([],[], lw=2, color=color)[0]
            lines.append(lobj)

        # initialization function: plot the background of each frame
        def init():    
            for line in lines:
                line.set_data([], [])
            return lines

        x = np.linspace(1, num_segments, num_segments) * self.segment_scaling * self.sim_params.simulation_segment_length() 
        x_label = f"Length ({self.segment_unit})"

        # the left plot is a moving window input plot
        axin_x = np.arange(len(P_LR_in)) * self.sim_params.simulation_time_step()
        axin.plot(axin_x * self.time_scaling, P_LR_in * self.power_scaling, color='blue')
        axin.set_xlabel(f"Time ({self.time_unit})")
        axin.set_ylabel(f"Input Power ({self.power_unit})")
        axin.set_ylim(0) 
        axin.invert_xaxis()  

        # the right plot is a moving window output plot
        # we plot the whole output and then show only part of it
        axout_x = np.arange(self.sim_params.nr_cycles) * self.sim_params.simulation_time_step()
        axout.plot(axout_x * self.time_scaling, P_LR_out * self.power_scaling, color='blue')
        axout.set_xlabel(f"Time ({self.time_unit})")
        axout.set_ylabel(f"Output Power ({self.power_unit})")
        axout.yaxis.tick_right()
        axout.yaxis.set_label_position("right")
        axout.set_ylim(0)
        # by inverting the axis, the rightmost datapoint of the center graph 
        # matches the leftmost datapoint of the output graph
        axout.invert_xaxis() 

        fig.tight_layout()
        # note that this assumes left and right graphs use same color code
        self.make_room_for_legend(ax, axin, axout)
        ax.legend(handles=lines, labels=["Optical Power L->R", "Optical Power R->L", "Carrier density"], 
                  bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=3)

        def animate(i):
            timestamp = self.video_start_t + i * self.frame_delta_t
            ax.set_xlabel(x_label + f' [time {fmt_eng(timestamp, digits=4)}s]')

            for lnum, line in enumerate(lines):        
                line.set_data(x, npdata[:,lnum,i])

            time_window = 500e-12 # TODO make this an input parameter
            axin.set_xlim((timestamp + time_window) * self.time_scaling, timestamp * self.time_scaling) 
            axout.set_xlim(timestamp * self.time_scaling, (timestamp - time_window) * self.time_scaling) 
            return lines
        
        # show a progress bar if needed - this is recommended as encoding can take a 
        # couple of second up to minutes for longer animations
        frames_iter = range(*frame_slice.indices(num_frames))
        if self.show_progress:
            # wrap iter with a tqdm to show a progress bar (tqdm is iterable too)
            frames_iter = tqdm(frames_iter, desc="writing frames", colour="red")

        # call the animator. blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames_iter, 
                                       interval=100, blit=True)

        # save the animation as an mp4.  This requires ffmpeg or mencoder to be
        # installed.  The extra_args ensure that the x264 codec is used, so that
        # the video can be embedded in html5.  You may need to adjust this for
        # your system: for more information, see
        # http://matplotlib.sourceforge.net/api/animation_api.html
        # anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

        if self.output_filename[-4:] != '.mp4':
            print(f"WARNING: output_filename '{self.output_filename}' does not end with .mp4.")

        # save movie to work_folder
        with chdir(self.work_folder):
            anim.save(self.output_filename, **self.anim_save_kwargs)

        if self.show_result:
            plt.show()


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

    frame_slice, frame_slice_count = _to_slice_and_num_frames(frame_range, nr_frames)  

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

    fig, (ax0, ax1)   = plt.subplots((2, 1))
    #ax    = plt.axes(xlim=(0, max_x), ylim=(0, max_y))
    ax0.set_xlim(0, max_x)
    ax0.set_ylim(0, max_y)

    lines = []
    for index,lay in enumerate(plotlays):
        lobj = ax0.plot([],[],lw=2,color=plotcols[index])[0]
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
        ax0.set_xlabel(x_label + f' [time {fmt_eng(timestamp, digits=4)}s]')

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


def _to_slice_and_num_frames(frame_range : FrameRange, nr_frames:int) -> tuple[slice, int]:
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

#################################################################################################
# The following functions are an attempt to create a still image of a laser animation
# - wireframe_plot plots the animation in a 3D projection plot
# - progressive_plot_2d squishes the y-axis of a single frame, so many can be stacked on top of
#   each other such that the y-axis doubles as time axis.
# 
# Both approaches have their downsides and I'm not super happy with the results so far. But I'm 
# leaving the code just in case it might be useful in the future.
#################################################################################################

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

    frame_slice, frame_slice_count = _to_slice_and_num_frames(frame_range, nr_frames)  

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

    frame_slice, frame_slice_count = _to_slice_and_num_frames(frame_range, nr_frames)  

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