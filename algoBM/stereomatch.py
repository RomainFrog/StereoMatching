import os
import sys
sys.path.append(os.getcwd())
import time
import cv2
from skimage import io
from skimage.color import rgb2gray
from blockmatching import *
from algoBM.utils import compute_census



def main(Ig, Id, output_file):
    start_time = time.time()
    
    Ig = cv2.imread(Ig, cv2.IMREAD_GRAYSCALE)
    Id = cv2.imread(Id, cv2.IMREAD_GRAYSCALE)
    
    assert Ig.shape == Id.shape, f"Left and right images dimensions must match, got {Ig.shape=} and {Id.shape=}"
    
    MAXDISP = 64
    N = 7
    N_MODE = 11
    HEIGHT = Ig.shape[0]
    WIDTH = Ig.shape[1]
    CSIZE=(3,3)
    
    #Preprocessing
    left_census, right_census = compute_census(Ig, Id, CSIZE, HEIGHT, WIDTH)
    
    # Processing
    disp = block_matching(left_census, right_census, N, MAXDISP, ZNSSD)
    disp = disp.astype(np.uint8)
    # Postprocessing
    disp = mode_filter(disp, N_MODE)
    disp = np.interp(disp, (0, np.max(disp)), (0, 255)).astype(np.uint8)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    
    io.imsave(output_file, disp)
    

if __name__ == '__main__':
    # arguments
    assert len(sys.argv) == 4, "Il faut 3 arguments : stereomatch.py im_gche.png im_dte.png disp_sortie.png"
    
    main(sys.argv[1], sys.argv[2], sys.argv[3])