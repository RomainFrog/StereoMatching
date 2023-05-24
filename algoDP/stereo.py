import cv2
import numpy as np
from numba import njit


@njit
def stereoMatching(leftImg,rightImg):
    rows = leftImg.shape[0]
    cols = leftImg.shape[1]
    
    # Matrices to store disparities : left and right
    leftDisp=np.zeros((rows,cols))
    rightDisp=np.zeros((rows,cols))
     
    occ = 20
    
    # Pick a row in the image to be matched
    for c in range (0,rows):
        # Cost matrix 
        colMat=np.zeros((cols,cols))
        
        # Disparity path matrix
        dispMat=np.zeros((cols,cols))
        
        # Initialize the cost matrix 
        for i in range(0,cols):
            colMat[i][0] = i*occ
            colMat[0][i] = i*occ
        
        # Iterate the row in both the images to find the path using dynamic programming
        # Progamme is similar to LCS(Longest common subsequence)
        
        for k in range (0,cols):
            for j in range(0,cols):        
                if(leftImg[c][k]>rightImg[c][j]):
                    match_cost=leftImg[c][k]-rightImg[c][j]
                else:
                    match_cost=rightImg[c][j]-leftImg[c][k]
                
                # Finding minimum cost    
                min1=colMat[k-1][j-1]+match_cost
                min2=colMat[k-1][j]+occ
                min3=colMat[k][j-1]+occ
                
                colMat[k][j]=cmin=min(min1,min2,min3)
                
                # Marking the path 
                if(min1==cmin):
                    dispMat[k][j]=1
                if(min2==cmin):
                    dispMat[k][j]=2
                if(min3==cmin):
                    dispMat[k][j]=3
        
        # Iterate the matched path and update the disparity value
        i=cols-1
        j=cols-1
        
        while (i!=0) and  (j!=0):
            if(dispMat[i][j]==1):
                leftDisp[c][i]=np.absolute(i-j)
                rightDisp[c][j]=np.absolute(j-i)
                i=i-1
                j=j-1
            elif(dispMat[i][j]==2):
                leftDisp[c][i]=0
                i=i-1
            elif(dispMat[i][j]==3):
                rightDisp[c][j]=0
                j=j-1
                
    left = np.interp(leftDisp, (leftDisp.min(), leftDisp.max()), (0, 255))
    right = np.interp(rightDisp, (rightDisp.min(), rightDisp.max()), (0, 255))
                
    return left, right