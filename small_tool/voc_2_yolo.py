import sys
sys.path.append(".")
import xml.etree.ElementTree as ET
import yaml
from tqdm import tqdm
from utils.general import download, Path

names =  [ '__background__', # always index 0
                         'bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck']


# names =  [ '__background__', # always index 0
#                          'bus', 'bicycle', 'car', 'motorbike', 'person', 'rider', 'train', 'truck']

def convert_label(path, lb_path, year, image_id):
    def convert_box(size, box):
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh

    in_file = open(path / f'VOC{year}/Annotations/{image_id}.xml')
    out_file = open(lb_path, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in names and not int(obj.find('difficult').text) == 1:
            xmlbox = obj.find('bndbox')
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
            cls_id = names.index(cls)  # class id
            out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')


# Download

# url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
# urls = [url + 'VOCtrainval_06-Nov-2007.zip',  # 446MB, 5012 images
#         url + 'VOCtest_06-Nov-2007.zip',  # 438MB, 4953 images
#         url + 'VOCtrainval_11-May-2012.zip']  # 1.95GB, 17126 images
# download(urls, dir=dir / 'images', delete=False)

# Convert
path = '/home/smtm/qhd/dataset/datasets/Rain'
dir = Path(path)  # dataset root dir
path =Path(path)
# for year, image_set in  ('2007', 'train_day'), ('2007', 'val_night'):
for year, image_set in  ('2007', 'val'), ('2007', 'val'):
    imgs_path = dir / 'images' / f'{image_set}{year}'
    lbs_path = dir / 'labels' / f'{image_set}{year}'
    imgs_path.mkdir(exist_ok=True, parents=True)
    lbs_path.mkdir(exist_ok=True, parents=True)

    image_ids = open(path / f'VOC{year}/ImageSets/Main/{image_set}.txt').read().strip().split()
    for id in tqdm(image_ids, desc=f'{image_set}{year}'):
        f = path / f'VOC{year}/JPEGImages/{id}.jpg'  # old img path
        lb_path = (lbs_path / f.name).with_suffix('.txt')  # new label path
        # f.rename(imgs_path / f.name)  # move image
        convert_label(path, lb_path, year, id)  # convert labels to YOLO format
