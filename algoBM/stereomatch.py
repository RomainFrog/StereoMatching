import os
import sys
import time
from skimage import io
from skimage.color import rgb2gray
from blockmatching import *

def main(Ig, Id, output_file):
    start_time = time.time()
    
    Ig = io.imread(Ig)
    Id = io.imread(Id)
    Ig = rgb2gray(Ig)
    Id = rgb2gray(Id)
    
    assert Ig.shape == Id.shape, f"Left and right images dimensions must match, got {Ig.shape=} and {Id.shape=}"
    
    maxdisp = 60
    N = 7
    disp = block_matching(Ig, Id, N, maxdisp, SAD)
    disp = mode_filter(disp, N)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    
    io.imsave(output_file, disp)
    
    

if __name__ == '__main__':
    # arguments
    assert len(sys.argv) == 4, "Il faut 3 arguments : stereomatch.py im_gche.png im_dte.png disp_sortie.png"
    
    main(sys.argv[1], sys.argv[2], sys.argv[3])