import cv2
import numpy as np
import re


# -1 depth 값 채우는 함수
def fill_invalid(depth, max_disp):
    row_length, column_length = depth.shape
    # occlude 위치 저장
    occluded_pixel = np.zeros((row_length, column_length))
    occluded_pixel[depth<0] = 1
    
    # left
    fillVals = np.ones((row_length)) * max_disp
    depth_left = depth.copy()
    for col in range(column_length):
        curCol = depth[:,col].copy()
        curCol[curCol==-1] = fillVals[curCol==-1] # 현재 열에서 occluded 된 위치 가져옴
        fillVals[curCol!=-1] = curCol[curCol!=-1] # 다음 열로 전달
        depth_left[:,col] = curCol
    
    # right
    fillVals = np.ones((row_length)) * max_disp
    depth_right = depth.copy()
    for col in reversed(range(column_length)):
        curCol = depth[:,col].copy()
        curCol[curCol==-1] = fillVals[curCol==-1]
        fillVals[curCol!=-1] = curCol[curCol!=-1]
        depth_right[:,col] = curCol

    filled_depth = np.fmin(depth_left, depth_right)
    return occluded_pixel, filled_depth


# -1 을 임시로 채운 depth 맵에 weighted median filter 적용, -1 픽셀 위치만 적용
def weighted_median_filter(left_img_origin, filled_depth, occluded_pixel, r_median, max_disp):
    sigma_c = 0.1
    sigma_s = 9

    row_length, column_length, _ = left_img_origin.shape
    filtered_img = cv2.medianBlur(left_img_origin, 3) # 이미지 median blur

    window_size = r_median // 2
    weighted_median = np.ones((row_length, column_length)) * -1

    for row in range(row_length):
        print('occlusion [%d/%d]' % (row, row_length))
        for col in range(column_length):
            if occluded_pixel[row, col] == 0: # occluede 아닌 픽셀은 무시함
                continue

            weights = [0.0 for _ in range(max_disp+1)]
            total_sum = 0.0
            # window 계산
            for i in range(max(row - window_size, 0), min(row + window_size, row_length)):
                for j in range(max(col - window_size, 0), min(col + window_size, column_length)):
                    # 현재 픽셀에서 멀리 떨어질수록 weight 적게 줌
                    spatial_diff = np.sqrt( np.square(row-i) + np.square(col-j) )
                    # color 값 변화가 크면 weight 를 적게줌
                    color_diff = np.sqrt( np.sum( np.square(filtered_img[row, col, :] - filtered_img[i, j, :]), axis=-1 )  )

                    weight = np.exp( - spatial_diff/(sigma_s*sigma_s) - color_diff/(sigma_c*sigma_c) ) # 정규화
                    weights[filled_depth[i, j]] += weight
                    total_sum += weight
            
            # 임계값을 넘어간 순간 occlude 픽셀을 해당 depth로 치환
            cum_sum = 0.0
            for i in range(max_disp+1):
                cum_sum += weights[i]
                if (cum_sum > total_sum / 2):
                    weighted_median[row, col] = i
                    break
    
    filled_depth[weighted_median!=-1] = weighted_median[weighted_median!=-1]
    return filled_depth


# pfm 파일 읽기
def read_pfm_file(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # test, scale = read_pfm_file('./piano-gt-l.pfm')
    # print(scale)

    # test = plt.imread('./dp_images/dp_piano_35_180_grad_fig15_1.png')
    # plt.imshow(test, cmap="jet")
    # plt.show()

    # save image in color
    # plt.imsave("color.png", test, cmap="jet")

    # 흑백 depth 맵 2014 데이터 처럼 컬러로
    images_home = './dp_images/'
    save_home = './color_dp_images'
    os.makedirs(save_home, exist_ok=True)
    file_list = os.listdir(images_home)
    for now_file in file_list:
        if 'fig' in now_file:
            continue
        test = plt.imread(os.path.join(images_home, now_file))
        plt.imsave(os.path.join(save_home, now_file), test, cmap="jet")