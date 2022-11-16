from utils.general import scale_coords
import torch 
import numpy as np
import cv2

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def xyxy2xywhn(x, w=640, h=640, img_origin = (1024,2048,3), clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    # if clip:
    #     clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    rw = img_origin[1]
    rh = img_origin[0]
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) /rw # x center w/img_origin[1]/ w
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) /rh  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) /rw  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) /rh  # height
    return y

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y

    # y[:, 0] = w * (x[:, 0] - x[:, 2] / 2)  # top left x
    # y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + 104 # top left y
    # y[:, 2] = w * (x[:, 0] + x[:, 2] / 2)  # bottom right x
    # y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + 104 # bottom right y
    return y

def test_xyxy(img, bbox,i,epoch):
    img = img[i]
    img =torch.squeeze(img)
    img = img.cpu().numpy()
    img = img*255
    bbox = bbox.cpu().detach().numpy()
    img = img.transpose(1, 2, 0) # HWC to CHW, BGR to RGB
    cv2.imwrite('./runs/teacher_img/xyxy' + str(epoch) +'.jpg', img)
    # x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    # xyxy1 = xywhn2xyxy(bbox, 416, 416, 0, 0)
    # xyxy = xyxy[0]
    # rr = int(xyxy[0])
    img1 = img.copy()
    # img1 = cv2.resize(img1,(2048,1024))
    for xyxy in bbox:
        cv2.rectangle(img1, (int(xyxy[1]), int(xyxy[2])), (int(xyxy[3]), int(xyxy[4])), (255,255, 0))
    cv2.imwrite('./runs/teacher_img/xyxybbox' + str(epoch) + '.jpg', img1)



def test_bbox(img, bbox,i,epoch):
    if len(bbox) > 0:
        img = img[i]
        img =torch.squeeze(img)
        img = img.cpu().numpy()
        img = img*255
        bbox = bbox.cpu().detach().numpy()
        img = img.transpose(1, 2, 0) # HWC to CHW, BGR to RGB
        cv2.imwrite('./runs/teacher_img/wid' + str(epoch) +'.jpg', img)
        # x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        xyxy1 = xywhn2xyxy(bbox, 416, 416, 0, 0)
        # xyxy = xyxy[0]
        # rr = int(xyxy[0])
        img1 = img.copy()
        for xyxy in xyxy1:
            cv2.rectangle(img1, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255,255, 0))
        cv2.imwrite('./runs/teacher_img/wid_bbox' + str(epoch) + '.jpg', img1)
    else:
        img = img[i]
        img =torch.squeeze(img)
        img = img.cpu().numpy()
        img = img*255
        img = img.transpose(1, 2, 0) # HWC to CHW, BGR to RGB
        cv2.imwrite('./runs/teacher_img/wid' + str(epoch) +'.jpg', img)

def labels2targets(labels,  img_train, img_origin, imgs_t, iter, epoch, targets, imgs, saveimg = False):
    
    for i,det in enumerate(labels):
        # 先转成输出的xyxy完整形式
        det[:, :4] = scale_coords(img_train.shape[2:], det[:, :4], (416,416,3)).round()
        # img_train_ = torch.tensor([416,208])
        # det[:, :4] = scale_coords(img_train_, det[:, :4], img_origin).round()
        det_cls_bbox = torch.cat([det[:, 5:],det[:, :4]], dim = 1)
        if iter == 100 and saveimg:
            test_xyxy(imgs_t,det_cls_bbox,i,epoch)
        # det_cls_bbox[:, 1:5] = xyxy2xywhn(det_cls_bbox[:, 1:5], w=img_train.shape[2], h=img_train.shape[3], img_origin = img_origin, clip=True, eps=1E-3)
        det_cls_bbox[:, 1:5] = xyxy2xywhn(det_cls_bbox[:, 1:5], w=208, h=416, img_origin = (416,416,3), clip=True, eps=1E-3)
        if iter == 100 and saveimg:
            test_bbox(imgs,targets[:, 2:6],i,epoch)
            test_bbox(imgs_t,det_cls_bbox[:, 1:5],i,epoch = (epoch+300))

        nl = len(det_cls_bbox)
        labels_out = torch.zeros((nl, 6))
        
        if nl:
            labels_out[:, 1:] = det_cls_bbox
            labels_out[:, 0] = torch.tensor(i, dtype=torch.float32)
        if i == 0:
            labels_out_all = labels_out
        if i > 0: 
            labels_out_all = torch.cat((labels_out_all,labels_out),0)
    # print("targets:",targets)
    # print("labels_out_all:",labels_out_all)
    return labels_out_all