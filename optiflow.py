import numpy as np
from cv2 import cv2
from itertools import product

def make_colorwheel():
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    
    # RY
    colorwheel[:RY, 0] = 255
    colorwheel[:RY, 1] = np.floor(np.arange(RY)*255/RY)#.reshape(-1, 1)
    col = RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(np.arange(YG)*255/YG)#.reshape(-1, 1)
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(np.arange(GC)*255/GC)#.reshape(-1, 1)
    col = GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(np.arange(CB)*255/CB)#.reshape(-1, 1)
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(np.arange(BM)*255/BM)#.reshape(-1, 1)
    col = BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(np.arange(MR)*255/MR)#.reshape(-1, 1)
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    img = np.zeros((*u.shape, 3))

    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u)/np.pi #
    fk = (a+1) / 2 * (ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(3):
        tmp = colorwheel[:, i]
        col0 = tmp[k0]/255
        col1 = tmp[k1%len(tmp)]/255
        col = (1-f)*col0 + f*col1
        idx = rad <= 1
        col[idx] = (1-rad[idx])*(1-col[idx])
        col[~idx] *= 0.75
        img[:, :, i] = np.floor(255*col*(1-nanIdx))

    return img

def flow_to_color(flow):
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    rad = np.sqrt(u**2+v**2)
    maxrad = np.max(rad)
    u /= maxrad + np.finfo(np.float32).eps
    v /= maxrad + np.finfo(np.float32).eps
    return compute_color(u, v)

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def L1Loss(img_1, img_2):
    return np.sum(np.abs(img_1.astype(np.int32) - img_2.astype(np.int32)))

ground_truth = cv2.imread('car_flow.jpg')
car_1 = cv2.cvtColor(cv2.imread('car1.jpg'), cv2.COLOR_BGR2GRAY)
car_2 = cv2.cvtColor(cv2.imread('car2.jpg'), cv2.COLOR_BGR2GRAY)
# cv2.imwrite('gray_car1.jpg', car_1)
# cv2.imwrite('gray_car2.jpg', car_2)

# pyr_scale = [0.5]
# levels = [1, 3, 5]
# winsize = [5, 10, 15, 20]
# iterations = [1, 3, 5, 7]
# poly = [(5, 1.1), (7, 1.5)]
# flags = [0, cv2.OPTFLOW_FARNEBACK_GAUSSIAN]
# cv2.calcOpticalFlowFarneback(car_1, car_2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
# for i in product(pyr_scale, levels, winsize, iterations, poly, flags):
#     print(i)
#     flow = cv2.calcOpticalFlowFarneback(car_1, car_2, None, i[0], i[1], i[2], i[3], i[4][0], i[4][1], i[5])
#     print(flow.shape)
#     cv2.imwrite('cv_flow_x.jpg', flow[:, :, 0]*30)
#     cv2.imwrite('cv_flow_y.jpg', flow[:, :, 1]*30)
#     flow = flow_to_color(flow)
#     cv2.imshow('flow', flow)
#     cv2.waitKey()
    
#     exit()
#     print(L1Loss(flow, ground_truth), i)

vx = cv2.imread('vx_show.jpg', 0).reshape(480, 640, 1)
vy = cv2.imread('vy_show.jpg', 0).reshape(480, 640, 1)
flow = np.concatenate((vx, vy), 2).astype(np.float32)
mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
hsv = np.zeros((480, 640, 3), dtype=np.uint8)
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
cv2.imshow('frame2',bgr)
k = cv2.waitKey()
flow = flow_to_color(flow)
cv2.imshow('flow', flow)
cv2.waitKey()