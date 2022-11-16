import cv2
import numpy as np
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.array([[0,0,0,0],[0,0,0,0]])
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.array([[0,0,0,0],[0,0,0,0]])
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)*rw  # top left x
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)*rh  # top left y
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)*rw  # bottom right x
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)*rh  # bottom right y
    return y

# 假设图像为test.jpg
img = cv2.imread('/home/smtm/qhd/code/code_git/yolov5-6.0/data/images/zidane.jpg')
h, w, _ = img.shape


# yolo标注数据文件名为786_rgb_0616.txt
# with open('786_rgb_0616.txt', 'r') as f:
# 	temp = f.read()
# 	temp = temp.split()
	# ['1', '0.43906', '0.52083', '0.34687', '0.15']

# 根据第1部分公式进行转换
# x_, y_, w_, h_ = 219, 223, 254, 373
# x_, y_, w_, h_ = 438, 422, 510, 720

# x1, y1, x2, y2 = 438, 422, 510, 720
rw = 1000/1280
rh = 600/700
xyxy = np.array([[438, 422, 510, 720],[0,0,0,0]])
xywh = xyxy2xywh(xyxy)
xyxy1 = xywh2xyxy(xywh)
xyxy1 = xyxy1[0]
x1, y1, x2, y2 = xyxy1[0],xyxy1[1],xyxy1[2],xyxy1[3]
# x1 = w * x_ - 0.5 * w * w_
# x2 = w * x_ + 0.5 * w * w_
# y1 = h * y_ - 0.5 * h * h_
# y2 = h * y_ + 0.5* h * h_
# 1280,720
img = cv2.resize(img, (1000, 600)) 
# 画图验证，注意画图坐标要转换成int格式
cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,255, 0))
cv2.imwrite('./wid.jpg', img)
cv2.waitKey(0)