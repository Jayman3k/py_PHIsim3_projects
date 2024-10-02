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


DEFAULT_FRAME_WARNING_TRESHOLD =  5000
DEFAULT_FRAME_ABORT_TRESHOLD   = 20000 

def animate_video(work_folder, output_filename, 
                  # optional arguments:
                  phisimout_name='PHIsimout.txt', 
                  show_progress=True, show_result=True, print_debug=True,
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

    if print_debug:
        print('Nr of frames:', nr_frames, 'Nr of data:', nr_sl_data)

    if sanity_limits:
        if warn_limit := sanity_limits.get("warn", None):
            if nr_frames > warn_limit:
                print(f'WARNING: video data has {nr_frames} frames, resulting file will be large and encoding will take a while')
        if err_limit := sanity_limits.get("error", None):
            if nr_frames > err_limit:
                print(f'ERROR: video data would have {nr_frames} frames, ABORTING encode') 
                return  
    
    # load the other data files with RLp en carriers as well
    video_dat_RLp = np.loadtxt(vid_dat_name_RLp, unpack=True, ndmin=2)
    video_dat_car = np.loadtxt(vid_dat_name_car, unpack=True, ndmin=2)

    max_LRp = video_dat_LRp.max()
    max_RLp = video_dat_RLp.max()
    max_car = video_dat_car.max()

    maxy = max_LRp
    if max_RLp > max_LRp:
        maxy = max_RLp   

    # copy data to a 3d array and scale all data to 1
    npdata = np.random.rand(nr_sl_data, 3, nr_frames)
    npdata[:,0,:] = video_dat_LRp[:,:] / maxy
    npdata[:,1,:] = video_dat_RLp[:,:] / maxy
    npdata[:,2,:] = video_dat_car[:,:] / max_car

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
        x = np.linspace(1, nr_sl_data, nr_sl_data)
        
        for lnum,line in enumerate(lines):        
            line.set_data(x, npdata[:,lnum,i])

        return lines
    
    # show a progress bar if needed - this is recommended as encoding can take a 
    # couple of second up to minutes for longer animations
    if show_progress:
        frames = tqdm(range(nr_frames), desc="writing frames", colour="red")
    else:
        frames = nr_frames

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, 
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