import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import imageio
import os
import argparse
from utils import fill_invalid, weighted_median_filter
import math


parser = argparse.ArgumentParser()
parser.add_argument('--left_image', '-l', default = './source_images/tsukuba-l.png', type=str, help='left image path')
parser.add_argument('--right_image', '-r', default = './source_images/Piano-r.png', type=str, help='right image path')
parser.add_argument('--save_image', '-s', default = './dp_images/dp_cone.png', type=str, help='save image path')
parser.add_argument('--occlusion_cost', '-o', default = 1000, type=int, help='occlusion cost for dp')
parser.add_argument('--block_size', '-p', default = 5, type=int, help='block size for dissimilarity')
parser.add_argument('--camera_baesline', '-c', default = 64, type=int, help='camera baseline, scan line consistency')
parser.add_argument('--match_cost', '-m', default = 'grad', type=str, help='type of cost computation [abs, sqr, ncc, grad, census]')
parser.add_argument('--filter', '-f', action='store_true') # store_true의 경우 default 값은 false이며, 인자를 적어 주면 true가 저장된다.
parser.add_argument('--filter_type', '-ft', default = 'gaussian', type=str, help='filter type [gaussian, bilateral, guided]')
parser.add_argument('--kernel_size', '-cs', default = 3, type=int, help='gaussian filter kernel size')
parser.add_argument('--filling', '-fl', action='store_true')
parser.add_argument('--fill_type', '-flt', default = 'wmf', type=str, help='occulsion filling type [simple, wmf(=weighted median filter)]')
parser.add_argument('--resize_scale', '-rs', default = 0.3, type=float, help='resize 2014 data set')
args = parser.parse_args()
print(args)
print()

# 이미지 저장 경로 생성
save_path = './dp_images'
os.makedirs(save_path, exist_ok=True)

filename, file_extension = os.path.splitext(args.save_image)
insert_name = '_%d_%d_%s' % (args.occlusion_cost, args.camera_baesline, args.match_cost)
save_image_name = filename + insert_name + file_extension
print('save image name:', save_image_name)

# 이미지 읽기
left_img_origin = cv2.imread(args.left_image)
right_img_origin = cv2.imread(args.right_image)
print('image shape:', left_img_origin.shape)

# 수행 속도때문에 2014 데이터 셋 크기 조절
if args.resize_scale != 1:
    # cv2.INTER_AREA, 영상 축소 시 디테일을 보존
    left_img_origin = cv2.resize(left_img_origin, (0, 0), fx=args.resize_scale, fy=args.resize_scale, interpolation=cv2.INTER_AREA)
    right_img_origin = cv2.resize(right_img_origin, (0, 0), fx=args.resize_scale, fy=args.resize_scale, interpolation=cv2.INTER_AREA)
    print(args.resize_scale, 'resized image shape:', left_img_origin.shape)

# 흑백 이미지
img_L = cv2.cvtColor(left_img_origin, cv2.COLOR_BGR2GRAY)
img_R = cv2.cvtColor(right_img_origin, cv2.COLOR_BGR2GRAY)


if args.filter:
    print('filter:', args.filter_type)
    if args.filter_type == 'gaussian':
        # Make laplacian filter
        # filter = np.array([[-1, -1, -1],
        #                 [-1,  8, -1],
        #                 [-1, -1, -1]])

        # Make 2D gaussian filter ( Option )
        filter = cv2.getGaussianKernel(ksize=args.kernel_size, sigma=1)
        filter = filter * filter.T

        # Apply filter to image
        img_L = cv2.filter2D(img_L, -1, filter)
        img_R = cv2.filter2D(img_R, -1, filter)

    elif args.filter_type == 'bilateral':
        d = 7           # d : 필터링에 이용하는 이웃한 픽셀의 지름을 정의 불가능한경우 sigmaspace 를 사용
        sigmaColor = 50 # sigmaColor : 컬러공간의 시그마공간 정의, 클수록 이웃한 픽셀과 기준색상의 영향이 커진다
        sigmaSpace = 50 # sigmaSpace : 시그마 필터를 조정한다. 값이 클수록 긴밀하게 주변 픽셀에 영향을 미친다. d>0 이면 영향을 받지 않고, 그 외에는 d 값에 비례한다. 
                        # https://eehoeskrap.tistory.com/125 

        img_L = cv2.bilateralFilter(img_L, d, sigmaColor, sigmaSpace)
        img_R = cv2.bilateralFilter(img_R, d, sigmaColor, sigmaSpace)
        # cv2.imshow('left', img_L)
        # cv2.imshow('right', img_R)
        # cv2.waitKey(0)

    elif args.filter_type == 'guided':
        r = 9
        eps = 0.01 # 작을수록 엣지가 살아나며 클수록 mean필터에 가까워짐

        img_L = cv2.ximgproc.guidedFilter(left_img_origin, img_L, r, eps)
        img_R = cv2.ximgproc.guidedFilter(right_img_origin, img_R, r, eps)
        # cv2.imshow('left', img_L)
        # cv2.imshow('right', img_R)
        # cv2.waitKey(0)


# Height and Width
row_length, column_length, channel_length = left_img_origin.shape

# Patch size
block_size = args.block_size
half_size = block_size // 2

# depth map
disparity_left = np.zeros(img_L.shape, np.int32)
disparity_right = np.zeros(img_L.shape, np.int32)

occlusion_cost = args.occlusion_cost
camera_baesline = args.camera_baesline
print('occlusion cost:', occlusion_cost)

# matching cost 변화에 따라 파라미터 조정
if args.match_cost == 'ncc':
    eps = 1e-6
elif args.match_cost == 'grad':
    # dissimilarity 미리 계산
    threshBorder = 3
    thresColor = 7
    thresGrad = 2
    gamma = 0.11

    # compute gradient from grayscale images
    left_gradient_x = np.gradient(img_L, axis=1)
    left_gradient_x = left_gradient_x + 128 # 음수 값 처리
    left_gradient_y = np.gradient(img_L, axis=0)
    left_gradient_y = left_gradient_y + 128

    right_gradient_x = np.gradient(img_R, axis=1)
    right_gradient_x = right_gradient_x + 128
    right_gradient_y = np.gradient(img_R, axis=0)
    right_gradient_y = right_gradient_y + 128

    cost_volume = np.ones((row_length, column_length, camera_baesline + 1)).astype(np.float32) * threshBorder

    # 0 ~ camera_baesline 에 해당하는 cost 계산
    for d in range(camera_baesline + 1):
        print('grad cost:', d)
        # color cost
        cost_temp = np.ones((row_length, column_length, channel_length)) * threshBorder
        cost_temp[:, d:, :] = right_img_origin[:, :column_length-d, :]
        cost_color = abs(left_img_origin - cost_temp)
        cost_color = np.mean(cost_color, axis = 2) # bgr mean
        cost_color = np.minimum(cost_color, thresColor)

        # gradient cost
        cost_temp = np.ones((row_length, column_length)) * threshBorder
        cost_temp[:, d:] = right_gradient_x[:, :column_length-d]
        cost_grad_x = abs(left_gradient_x - cost_temp)

        cost_temp = np.ones((row_length, column_length)) * threshBorder
        cost_temp[:, d:] = right_gradient_y[:, :column_length-d]
        cost_grad_y = abs(left_gradient_y - cost_temp)

        cost_grad = np.minimum(cost_grad_x + cost_grad_y, thresGrad)
        
        cost_volume[:, :, d] = gamma * cost_color + (1 - gamma) * cost_grad

    # depth_left = np.argmin(cost_volume, axis = 2)
    # imageio.imwrite('./grad_test.png', depth_left)
    # exit(0)
elif args.match_cost == 'census':
    # dissimilarity 미리 계산 - absolute difference 만, census 는 for 문에서
    lambdaAd = 10.
    lambdaCensus = 5.

    cost_ad = np.ones((row_length, column_length, camera_baesline + 1)).astype(np.float32)

    # 0 ~ camera_baesline 에 해당하는 cost 계산
    for d in range(camera_baesline + 1):
        print('grad cost:', d)
        # color cost
        cost_temp = np.ones((row_length, column_length, channel_length))
        cost_temp[:, d:, :] = right_img_origin[:, :column_length-d, :]
        cost_temp = abs(left_img_origin - cost_temp)
        cost_temp = np.mean(cost_temp, axis = 2) # bgr mean
        cost_ad[:, :, d] = cost_temp

    # depth_left = np.argmin(cost_ad, axis = 2)
    # imageio.imwrite('./grad_test.png', depth_left)
    # exit(0)


for row_idx in range(half_size, row_length - half_size):
    print('depth [%d / %d]' % (row_idx, row_length - half_size))
    dp = np.zeros((column_length, column_length))
    back_track = np.ones((column_length, column_length))

    for i in range(column_length):
        dp[0, i] = i * occlusion_cost
    for i in range(column_length):
        dp[i, 0] = i * occlusion_cost

    for i in range(half_size, column_length - half_size): # right image
        mask_R = img_R[row_idx-half_size:row_idx+half_size+1, i-half_size:i+half_size+1]
        
        for j in range(i, min(i + camera_baesline + 1, column_length - half_size)): # left image - scan line consistency
            mask_L = img_L[row_idx-half_size:row_idx+half_size+1, j-half_size:j+half_size+1]

            if args.match_cost == 'sqr':
                dissimilarity = np.sum((mask_R - mask_L) ** 2)
            elif args.match_cost == 'abs':
                dissimilarity = np.sum(abs(mask_R - mask_L))
            elif args.match_cost == 'ncc':
                mask_R = (mask_R - np.mean(mask_R)) / (np.std(mask_R) + eps)
                mask_L = (mask_L - np.mean(mask_L)) / (np.std(mask_L) + eps)
                dissimilarity = -np.sum(mask_R * mask_L) # 음수로 만듦, 값이 작을수록 좋다
            elif args.match_cost == 'grad':
                dissimilarity = np.sum(cost_volume[row_idx-half_size:row_idx+half_size+1, j-half_size:j+half_size+1, j-i])
            elif args.match_cost == 'census':
                p_ad = np.sum(cost_ad[row_idx-half_size:row_idx+half_size+1, j-half_size:j+half_size+1, j-i])

                # 중앙 값보다 작으면 0, 아니면 1
                mask_R = np.where(mask_R < mask_R[half_size, half_size], 0, 1)
                mask_L = np.where(mask_L < mask_L[half_size, half_size], 0, 1)
                p_census = np.sum( np.logical_xor(mask_R, mask_L) )
                dissimilarity = (1 - math.exp(-p_ad / lambdaAd)) + (1 - math.exp(-p_census / lambdaCensus))
                # print(dissimilarity)

            # 세방향 값 계산
            min1 = dp[i - 1, j - 1] + dissimilarity
            if j == i + camera_baesline: # left 가 camera baseline 보다 넘어서게 차이날 수는 없음, 대각선 위로 넘어가면 안됨
                min2 = 100000000
            else:
                min2 = dp[i - 1, j] + occlusion_cost
            if i == j: # left 가 right 보다 뒤로 갈 수는 없음, 대각선 아래로 넘어가면 안됨
                min3 = 100000000
            else:    
                min3 = dp[i, j - 1] + occlusion_cost
            cmin = min(min1, min2, min3)
            dp[i, j] = cmin # cost
            if cmin == min1:
                back_track[i, j] = 1
            elif cmin == min2:
                back_track[i, j] = 2
            elif cmin == min3:
                back_track[i, j] = 3

    i = column_length - 1
    j = column_length - 1

    while i != 0 and j != 0:
        assert i <= j and j <= i + camera_baesline
        if back_track[i, j] == 1:
            disparity_right[row_idx, i] = abs(i - j) # disparity
            disparity_left[row_idx, j] = abs(i - j) # disparity
            i = i - 1
            j = j - 1
        elif back_track[i, j] == 2:
            i = i - 1
        elif back_track[i, j] == 3:
            j = j - 1
    
    # if row_idx == row_length // 3:
    #     break


if args.filling:
    print('occlusion:', args.fill_type)
    if args.fill_type == 'simple':
        # handle occlusion
        depth_temp = np.zeros(disparity_left.shape)
        for i in range(row_length):
            print('occlusion [%d / %d]' % (i, row_length))
            for j in range(column_length):
                # depth 가 0 이라면 좌우 가까운 값을 가져옴
                if disparity_left[i, j] == 0:
                    to_left = j - 1
                    to_right = j + 1
                    while to_left >= 0 or to_right < column_length:
                        if to_left >= 0 and disparity_left[i, to_left] != 0:
                            depth_temp[i, j] = disparity_left[i, to_left]
                            break
                        if to_right < column_length and disparity_left[i, to_right] != 0:
                            depth_temp[i, j] = disparity_left[i, to_right]
                            break
                        to_left -= 1
                        to_right += 1

        for i in range(row_length):
            print('occlusion copy [%d / %d]' % (i, row_length))
            for j in range(column_length):
                if depth_temp[i, j] != 0:
                    disparity_left[i, j] = depth_temp[i, j]

    elif args.fill_type == 'wmf':
        r_median = 15
        
        depth = disparity_left.copy()
        # left-right consistancy check
        for row in range(row_length):
            for col in range(column_length):
                left_depth = disparity_left[row, col]
                # index 범위
                if left_depth > col:
                    continue

                right_depth = disparity_right[row, col-left_depth]
                # 왼쪽, 오른쪽 1 이상 차이나면 다시 계산
                if abs(left_depth - right_depth) >= 1:
                    depth[row, col] = -1

        occluded_pixel, filled_depth = fill_invalid(depth, camera_baesline)
        # imageio.imwrite('./filled_test.png', filled_depth)
        disparity_left = weighted_median_filter(left_img_origin, filled_depth, occluded_pixel, r_median, camera_baesline)    


# save image
imageio.imwrite(save_image_name, disparity_left)


# plot 여러개
# plt.subplot(121)
# plt.imshow(disparity_left, cmap='gray'); plt.axis('off')
# plt.xticks([]), plt.yticks([])

# plt.subplot(122)
# plt.imshow(disparity_right, cmap='gray'); plt.axis('off')
# plt.show()