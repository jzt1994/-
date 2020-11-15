import numpy as np
import cv2

from imutils import perspective
from skimage.filters import threshold_local
import cv2
import imutils
import numpy as np

def SetPoints(windowname, img):
    """
    输入图片，打开该图片进行标记点，返回的是标记的几个点的字符串
    """
    print('(提示：单击需要标记的坐标，Enter确定，Esc跳过，其它重试。)')
    points = []

    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(temp_img, (x, y), 10, (102, 217, 239), -1)
            points.append([x, y])
            cv2.imshow(windowname, temp_img)

    temp_img = img.copy()
    cv2.namedWindow(windowname)
    cv2.imshow(windowname, temp_img)
    cv2.setMouseCallback(windowname, onMouse)
    key = cv2.waitKey(0)
    if key == 13:  # Enter
        print('坐标为：', points)
        del temp_img
        cv2.destroyAllWindows()
        return points
    elif key == 27:  # ESC
        print('跳过该张图片')
        del temp_img
        cv2.destroyAllWindows()
        return
    else:
        print('重试!')
        return SetPoints(windowname, img)

def down_scale(image):
    height, width, _ = image.shape
    new_dim = (int(width / 2), int(height / 2))  # (width, height)
    new_img = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
    return new_img


def add_mask(points,image):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # roi_points = np.array([[0, 360], [0, 298], [298, 205], [373, 217], [640, 329], [640, 360]], dtype=np.int32)
    roi_points = np.array(points, dtype = np.int32)
    cv2.fillConvexPoly(mask, roi_points, 255)
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    return masked_img


def dense_optical_flow(prev_img, image):
    bgr = None
    hsv = np.zeros_like(image)  # hsv.shape:(360, 640, 3)
    hsv[..., 1] = 255  # color scale
    if prev_img is None:
        print('Initializing a prev_image...')
        prev_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # prev_img.shape:(360, 640)
    else:
        next_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_img, next_img, None,
                                            pyr_scale=0.5,
                                            levels=3,
                                            winsize=15,
                                            iterations=3,
                                            poly_n=5,
                                            poly_sigma=1.1,
                                            flags=0)  # flow.shape:(360, 640, 2)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2  # Hue, corresponds to direction
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value -> Magnitude
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        prev_img = next_img
    return bgr, prev_img


def extra_point(aol,img):
    points = SetPoints(str(aol), img)
    temp = np.array(points)
    pts = temp.reshape(4, 2).astype(np.float32)
    return pts

def read_img(path):
    image = cv2.imread(path)
    w=image.shape[0]
    h=image.shape[1]
    ratio = image.shape[0] / 500.0  # 比例
    orig = image.copy()
    image = imutils.resize(image, height=500)
    return image

def extra_point(aol,img):
    points = SetPoints(str(aol), img)
    temp = np.array(points)
    pts = temp.reshape(4, 2).astype(np.float32)
    return pts

def extra_point_lots(aol,img):
    points = SetPoints(str(aol), img)
    temp = np.array(points)
    pts = np.float32(temp).reshape(-1,1,2)
    return pts

def main():
    image1 = cv2.imread("002.png")
    image2 = read_img("drone.png")
    
    point_show = SetPoints('real',image1)
    image1 = add_mask(point_show, image1)
    
    # point1 = extra_point('real',image1)
    # point2 = extra_point("sat", image2)

    point1 = extra_point_lots('real',image1)
    point2 = extra_point_lots("drone", image2)
    
    
    #找到投影变换矩阵
    M, mask = cv2.findHomography(point1, point2, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    # #进行投影变换
    # h,w,d = image1.shape
    # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    # out_img = cv2.perspectiveTransform(pts,M)
    # #实际坐标点和提取的角点必须一一对应呀，
    # M = cv2.getPerspectiveTransform(point1,point2)
    out_img = cv2.warpPerspective(image1,M,(image1.shape[0],700))
    # dst=cv2.perspectiveTransform(point2.reshape(1,4,2), M)
     
    cv2.imshow("Original", image1)
    cv2.imshow("Scanned",cv2.resize(out_img,(image1.shape[0],700)))
    
if __name__ == "__main__":
    main()

