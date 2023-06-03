import cv2
import numpy as np
from numba import njit


@njit
def stereoMatching(leftImg,rightImg, occ):
    rows = leftImg.shape[0]
    cols = leftImg.shape[1]
    
    # Matrices to store disparities : left and right
    leftDisp=np.zeros((rows,cols))
    rightDisp=np.zeros((rows,cols))
     
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


@njit
def mode_filter(img, N=5):
    h, w = img.shape
    filtered_img = np.zeros_like(img)

    margin = N//2
    # Parcourir chaque pixel de la carte de disparités
    for i in range(h):
        for j in range(w):
            # Extraire la fenêtre 5x5 centrée sur le pixel actuel
            block = img[max(0, i - margin):min(h, i + margin+1), max(0, j - margin):min(w, j + margin+1)]

            occ = {val:0 for val in set(block.flatten())}
            for val in block.flatten():
                occ[val]+=1
                
            max_k=0
            max_val=0
            for key, value in occ.items():
                if value > max_val:
                    max_k=key
                    max_val=value
            filtered_img[i,j]=max_k

    return filtered_img

from tqdm import tqdm
def stereo_2(original_Left_Image, original_Right_Image, occlusion):
    
    leftImage = np.asarray(original_Left_Image, dtype = np.float64)
    rightImage = np.asarray(original_Right_Image, dtype = np.float64)
    
        
    rowCount, columnCount = leftImage.shape
    
    costMatrix = np.empty([columnCount, columnCount], dtype=np.float64)
    M = np.empty([columnCount, columnCount], dtype=np.float64)
    disparity_Map_Left = np.empty([rowCount, columnCount], dtype=np.float64)
    disparity_Map_Right = np.empty([rowCount, columnCount], dtype=np.float64)

    cost = 0.0
    costMatrix[0, 0] = 0
    for row in tqdm(range(0, rowCount)):
    
        
        for i in range(1, columnCount):
            costMatrix[i, 0] = i * occlusion
            costMatrix[0, i] = i * occlusion
            
        for i in range(0, columnCount):
            for j in range(0, columnCount):

                # Cost function for matching features in the left and right images
                cost = abs(leftImage[row, i] - rightImage[row, j])
                
                min1 = costMatrix[i-1, j-1] + cost
                min2 = costMatrix[i-1, j] + occlusion
                min3 = costMatrix[i, j-1] + occlusion
        
                cmin = min(min1, min2, min3)

                costMatrix[i, j] = cmin
                
                # Forming path matrix
                if(cmin == min1):
                    M[i, j] = 1
                if(cmin == min2):
                    M[i, j] = 2
                if(cmin == min3):
                    M[i, j] = 3

        p = columnCount - 1
        q = columnCount - 1
        
        while(p != 0 and q !=0):
            
            # if feature in left and right image matches
            if(M[p, q] == 1):
                disparity_Map_Left[row, p] = abs(p-q)
                disparity_Map_Right[row, q] = abs(q-p)                
                p = p - 1
                q = q - 1
                
            # if feature in left image is occuluded
            elif(M[p, q] == 2):
                disparity_Map_Left[row, p] = 0
                p = p - 1
        
            # if feature in right image is occuluded
            elif(M[p, q] == 3):
                disparity_Map_Right[row, q] = 0
                q = q - 1
                
        costMatrix = np.empty([columnCount, columnCount], dtype=np.float64)
        M = np.empty([columnCount, columnCount], dtype=np.float64)
        
    return disparity_Map_Left, disparity_Map_Right