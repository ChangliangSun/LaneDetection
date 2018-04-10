import json
import cv2
import numpy as np
import math
from cv2 import VideoWriter,VideoWriter_fourcc,imread,resize

fps = 20
fourcc = VideoWriter_fourcc(*"MJPG")
videoWriter = cv2.VideoWriter("test__.avi", fourcc, fps, (1280,720))
transform = np.float32([[ -3.36525864e-01,-2.38611758e+00,9.48230640e+02],
                        [ -1.66679133e-02,-2.40999812e+00,8.27194741e+02], [ -2.81596111e-05,-3.30583419e-03,1.00000000e+00]])

#################### read image and json ########################
for img_index in range(1):
    # print(img_index+1)
    img = cv2. imread('img_%08d.png'%(img_index+1))
    #######################get road mask from deep learning result###############
    f = open("annotation_frame_%08d.json"%(img_index+1), encoding='utf-8')
    test_json = json.load(f)
    objs = test_json['objects']
    h = test_json['imgHeight']
    w = test_json['imgWidth']
    bg_img = np.ones((h,w),dtype='uint8')
    road_img = np.zeros((h,w),dtype='uint8')
    for labelInd in range(len(objs)):
        label = objs[labelInd]
        class_name = label['label']
        polygons = label['polygons']
        bimg = np.zeros((h,w),dtype='uint8')
        for index_cont in range(len(polygons)):
            cv2.fillPoly(bimg, np.array([polygons[index_cont]]), 1)
        if class_name == 'road':
            road_img = bimg
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
            bimg = cv2.dilate(bimg, kernel)
            bg_img[bimg==1] = 0
    road_img = road_img*bg_img
    ########################perspect image to bird-eye###########################
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.warpPerspective(img_gray,transform, (w,h), cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    road_img = cv2.warpPerspective(road_img,transform, (w,h), cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    ########################dyn threshold###########################
    img_temp = cv2.blur(img_gray,(39,1))
    img_gray = img_gray/255.0
    img_temp = img_temp/255.0
    img_temp = img_gray - img_temp
    ##########select regions by dyn threshold and gray threshold############
    thre_dyn = 0.05
    thre_gray = 120/255.0
    img_temp[img_temp<thre_dyn] = 0
    img_temp[img_temp>=thre_dyn] = 1
    img_temp[img_gray<=thre_gray] = 0
    img_temp = img_temp*road_img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_temp = cv2.morphologyEx(img_temp, cv2.MORPH_CLOSE, kernel)
    ##########contours select by contour features###########
    _,contours,__ = cv2.findContours((img_temp*255).astype(np.uint8), 1, 2)
    res_contours = []
    thre_area = 300
    thre_phi = 70
    for i in range(len(contours)):
        contour = contours[i]
        M = cv2.moments(contour)
        if M['m00']>thre_area:
            a = M['m10'] / M['m00']
            b = M['m01'] / M['m00']
            M11 = M['m11']/M['m00']-a*b
            M20 = M['m20']/M['m00']-a*a
            M02 = M['m02']/M['m00']-b*b
            phi = (-0.5*math.atan2(2*M11,M20-M02))*(180/np.pi)
            a = M20+M02
            b = math.sqrt(math.pow(M20-M02,2)+4*math.pow(M11,2))
            ra = math.sqrt(8*(a+b))/2.0
            rb = math.sqrt(8*(a-b))/2.0
            if abs(phi)>=thre_phi and rb<=10:
                res_contours.append(contour)
    img_lane_candidate = np.zeros((h, w), dtype='float32')
    cv2.fillPoly(img_lane_candidate, np.array(res_contours), 1.0)
    ##########histogram each column###########
    histogram = []
    for hist_index in range(w):
        histogram.append(np.sum(img_lane_candidate[:,hist_index]))
    ##########select the columns of the lanes###########
    lane_cols = []
    lane_count = 0
    thre_lane_dist = 220
    while 1:
        if max(histogram)==0:
            break
        lane_cols.append(histogram.index(max(histogram)))
        left = max([lane_cols[lane_count]-thre_lane_dist,0])
        right = min([lane_cols[lane_count]+thre_lane_dist,w-1])
        histogram[left:right+1] = [0 for i in range(right-left)]
        lane_count += 1
    print(img_index,lane_cols)
    ##########fit the candidate lanes to line and draw it###########
    transform_I = np.mat(transform).I
    thre_lane_width = 20
    img_lane_lines = np.zeros((len(lane_cols), h, w), dtype='float32')
    for index_lane in range(len(lane_cols)):
        left = max([lane_cols[index_lane]-thre_lane_width,0])
        right = min([lane_cols[index_lane]+thre_lane_width,w-1])
        img_lane_lines[index_lane,:,left:right] = 1
        img_lane_lines[index_lane] = img_lane_lines[index_lane]*img_lane_candidate
        img_temp = cv2.warpPerspective(img_lane_lines[index_lane], transform_I, (w, h),
                              cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        line_points = []
        line_map = np.zeros((h, w), dtype='float32')
        for i in range(h - 1, int(h / 3), -2):
            r_i = img_temp[i, :]
            nonzero_index = np.matrix(r_i).nonzero()
            nonzero_index = np.array(nonzero_index)
            if len(nonzero_index[0]) > 0:
                max_nonzero_index = max(nonzero_index[1])
                min_nonzero_index = min(nonzero_index[1])
                line_points.append([int((max_nonzero_index + min_nonzero_index) / 2), i])
        if len(line_points) > 3:
            [vx, vy, x, y] = cv2.fitLine(np.array(line_points), cv2.DIST_L2, 0, 0.01, 0.01)
            if abs(math.atan2(vy,vx)*(180.0/np.pi))<89:
                lefty = int((-x * vy / vx) + y)
                righty = int(((w - x) * vy / vx) + y)
                cv2.line(line_map, (w - 1, righty), (0, lefty), 1.0, 2)
                line_map[0:int(h / 2.2), :] = 0
        img[:, :, 0][line_map > 0] = 255
        img[:, :, 1][line_map > 0] = 0
        img[:, :, 2][line_map > 0] = 0
    ##########draw the candidate mask###########
    img_temp = cv2.warpPerspective(img_lane_candidate,transform_I, (w,h), cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    img_temp[img_temp>0] = 1.0
    img[:,:,0][img_temp>0] = 0
    img[:,:,1][img_temp>0] = 0
    img[:,:,2][img_temp>0] = 255

    cv2.imshow('res', img)
    cv2.waitKey(0)
    # videoWriter.write(img)
videoWriter.release()
cv2.destroyAllWindows()