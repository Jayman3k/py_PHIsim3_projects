# -*- coding: utf-8 -*-
"""
Adapted from scripts provided by Erwin Bente (e.a.j.m.bente@tue.nl), 
which were packaged with PHIsim (https://sites.google.com/tue.nl/phisim/home),
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
import sys                              # imports system package to manipulate files
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from tqdm import tqdm

from tools.PHIsim_sim_params import PHIsim_SimulationParams
from tools.fmt_utils import fmt_eng

DEFAULT_FRAME_WARNING_TRESHOLD =  2000
DEFAULT_FRAME_ABORT_TRESHOLD   = 20000 

def animate_video_slice(work_folder, output_filename, 
                        sim_params: PHIsim_SimulationParams,
                        time_window: tuple[float, float], 
                        **kwargs):
    """Animate a time slice of the simulation run by PHIsim. 

    For description of parameters, see animate_video()
    """
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

    animate_video(work_folder, output_filename, 
                  timestamp_info=(video_start, frame_delta_t),
                  frame_range=(frame_start, frame_end), 
                  **kwargs)


def animate_video(work_folder, output_filename, 
                  timestamp_info : tuple[float, float],
                  # optional arguments:
                  phisimout_name='PHIsimout.txt', 
                  show_progress=True, show_result=True, print_debug=True,
                  frame_range: tuple[int, int]=(None, None),
                  sanity_limits={"warn" : DEFAULT_FRAME_WARNING_TRESHOLD, 
                                 "error" : DEFAULT_FRAME_ABORT_TRESHOLD},
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

    PHIsimout_prefix = phisimout_name[:-4] # strip '.txt' the same way PHIsim does

    vid_dat_name_LRp  = f'{PHIsimout_prefix}_vid_LRp.txt'
    vid_dat_name_RLp  = f'{PHIsimout_prefix}_vid_RLp.txt'
    vid_dat_name_car  = f'{PHIsimout_prefix}_vid_carriers.txt'

    # store current work-dir, to restore at the end
    previous_work_dir = os.getcwd()
    os.chdir(work_folder)  # change the working directory

    video_dat_LRp = np.loadtxt(vid_dat_name_LRp, unpack=True, ndmin=2)
    # print(np.shape(video_dat_LRp))
    dimens        = np.shape(video_dat_LRp)
    nr_frames     = dimens[1]
    nr_sl_data    = dimens[0]
    
    # load the other data files with RLp en carriers as well
    video_dat_RLp = np.loadtxt(vid_dat_name_RLp, unpack=True, ndmin=2)
    video_dat_car = np.loadtxt(vid_dat_name_car, unpack=True, ndmin=2)

    max_LRp = video_dat_LRp.max()
    max_RLp = video_dat_RLp.max()
    max_car = video_dat_car.max()

    maxy = max_LRp
    if max_RLp > max_LRp:
        maxy = max_RLp 

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
    npdata[:,0,frame_slice] = video_dat_LRp[:,frame_slice] / maxy
    npdata[:,1,frame_slice] = video_dat_RLp[:,frame_slice] / maxy
    npdata[:,2,frame_slice] = video_dat_car[:,frame_slice] / max_car

    max_y = 1.0
    # max_y = video_dat_car.max()
    if print_debug:
        print('Maximum value in video data: ', max_y)

    #colours of the lines
    plotlays, plotcols = [0,1,2], ["red","blue","green"]

    # First set up the figure, the axis, and the plot element we want to animate
    fig   = plt.figure()
    ax    = plt.axes(xlim=(0, nr_sl_data), ylim=(0, max_y))
    y     = np.linspace(1, nr_sl_data, nr_sl_data)

    lines = []
    for index,lay in enumerate(plotlays):
        lobj = ax.plot([],[],lw=2,color=plotcols[index])[0]
        lines.append(lobj)

    # initialization function: plot the background of each frame
    def init():    
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(i):
        timestamp = timestamp_info[0] + i * timestamp_info[1]
        fig.get_axes()[0].set_xlabel(f'Segments (#) [time {fmt_eng(timestamp, digits=4)}s]')
        x = np.linspace(1, nr_sl_data, nr_sl_data)
        
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

    anim.save(output_filename, **kwargs)

    # restore previous work_dir
    os.chdir(previous_work_dir)

    if show_result:
        plt.show()


def __to_slice_and_num_frames(frame_range, nr_frames):
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