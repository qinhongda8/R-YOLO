# PyTorch lib
import argparse
import os

import cv2
# Tools lib
import numpy as np
import torch
from torch.autograd import Variable

# Models lib
# Metrics lib
# from metrics import calc_psnr, calc_ssim
# from models import *
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.det_conv0 = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.det_conv_mask = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.diconv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2),
            nn.ReLU()
        )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 4, dilation=4),
            nn.ReLU()
        )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 8, dilation=8),
            nn.ReLU()
        )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 16, dilation=16),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU()
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.outframe1 = nn.Sequential(
            nn.Conv2d(256, 3, 3, 1, 1),
            nn.ReLU()
        )
        self.outframe2 = nn.Sequential(
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, input,device):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        mask = Variable(torch.ones(batch_size, 1, row, col)).to(device) / 2.
        h = Variable(torch.zeros(batch_size, 32, row, col)).to(device)
        c = Variable(torch.zeros(batch_size, 32, row, col)).to(device)
        mask_list = []
        attention_map = []
        
        x = self.conv1(input)
        res1 = x
        x = self.conv2(x)
        x = self.conv3(x)
        res2 = x
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.conv7(x)
        x = self.conv8(x)
        frame1 = self.outframe1(x)
        x = self.deconv1(x)
        if x.shape != res2.shape:
            print('!ok')
        x = x + res2
        x = self.conv9(x)
        frame2 = self.outframe2(x)
        x = self.deconv2(x)
        x = x + res1
        x = self.conv10(x)
        x = self.output(x)
        x = input + x
        return mask_list, frame1, frame2, attention_map, x


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    parser.add_argument("--weight",type=str)
    parser.add_argument("--Normal_feak",type=str)
    args = parser.parse_args()
    return args


def align_to_four(img):
    # print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    # align to four
    a_row = int(img.shape[0] / 4) * 4
    a_col = int(img.shape[1] / 4) * 4
    img = img[0:a_row, 0:a_col]
    # print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    return img


def  predict(image):
    image = np.array(image, dtype='float32') / 255.
    mean=(0.406, 0.456, 0.485)
    std=(0.225,0.224, 0.229)
    mean = np.array(mean, dtype='float32')
    std = np.array(std, dtype='float32')
    image = image - mean
    image = image /std
    image = image[:, :, (2, 1, 0)]
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    image = Variable(image).to(device)

    out = model(image,device)[-1]

    out = out.cpu().data
    out = out.numpy()
    out = out.transpose((0, 2, 3, 1))
    out = out[0, :, :, :]
    out = out[:, :, (2, 1, 0)]
    out = out * std
    out = out + mean
    out = out * 255.
    return out





if __name__ == '__main__':
    args = get_args()
    if args.mode == "normal_to_adverse":
        # args.input_dir = './dataset/Normal_to_Foggy/images/Normal_train/'  # normal image 2975
        args.output_dir ='./dataset/Normal_to_Foggy/images/Normal_feak/'
        args.name_list = './Normal_feak.txt'
    elif  args.mode == "adverse_to_normal":
        # args.input_dir = './dataset/Normal_to_Foggy/images/Foggy_train/'  # 
        args.output_dir ='./dataset/Normal_to_Foggy/images/Foggy_feak/'
        args.name_list = './Foggy_feak.txt'
    path_to_output_dir = os.path.join(args.output_dir)
    path_weight = os.path.join(args.weight)
    os.makedirs(path_to_output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Generator().to(device)
    model.load_state_dict(torch.load(path_weight))


    input_list = sorted(os.listdir(args.input_dir))
    num = len(input_list)
    with open (args.name_list, 'a') as tx:
        for i in range(num):
            print('Processing image: %s' % (input_list[i]))
            img = cv2.imread(args.input_dir + input_list[i])
            original_size=img.shape
            # img = align_to_four(img)
            dsize = (416, 416)
            img = cv2.resize(img, dsize)
            result = predict(img)
            size = (original_size[1], original_size[0])
            result = cv2.resize(result, size)
            # img_name = input_list[i].split('.')[0]
            img_name = input_list[i]
            tx.write('source_' + img_name[:-4])
            tx.write('\n')
            # cv2.imwrite(args.output_dir + 'source_' + img_name[:-4] + '_fake_B.png', result)
            cv2.imwrite(args.output_dir + 'source_' + img_name[:-4] + '_fake_B.png', result)
           
        tx.close()