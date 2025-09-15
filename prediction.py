import os
import sys
import cv2
import argparse
import torch
from config import cfg

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
from PIL import Image
from Utils.visualize import label2color
from model import get_segmentation_Model

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
# model and dataset
parser.add_argument('--model', type=str, default='UCMNet',
                        choices=['LCAFormer'],
                        help='model name (default: fcn32s)')
parser.add_argument('--dataset', type=str, default='shvpli',
                    choices=['bijie', 'landslide4sense'],
                    help='dataset name (default: pascal_voc)')
args = parser.parse_args()


def demo(args, test_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    best_filename = '{}_{}_best_model.pth'.format(args.model, args.dataset)

    # best_filename = '{}_{}_{}.pth'.format(args.model, args.backbone, args.dataset)

    best_weight_path = os.path.join(cfg.DATA.WEIGHTS_PATH, best_filename)
    output_dir = os.path.join(cfg.DATA.PRE_PATH, args.model, args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('finish make output_dir done')

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])
    # input_pic = 'data/test/images/303.png'
    print('********')

    model = get_segmentation_Model(name=args.model, nclass=cfg.TRAIN.CLASSES).to(device)
    print(best_weight_path)
    model_dict = torch.load(best_weight_path)
    model.load_state_dict(model_dict)

    print('Finished loading model!')
    for _, path in enumerate(os.listdir(test_path)):
    # for i in range(len(os.listdir(test_path))):
        print(path)
        input_pic = os.path.join(test_path, path)
        image = Image.open(input_pic)

        images = transform(image).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output= model(images)
        pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
        # print(pred.shape)
        # print(pred)
        mask = label2color(cfg.TRAIN.CLASSES, pred)
        outname = os.path.splitext(os.path.split(input_pic)[-1])[0] + '.png'
        im = Image.fromarray(mask)
        im.save(os.path.join(output_dir, outname))
        print(os.path.join(output_dir, outname))
        # cv2.imwrite(os.path.join(output_dir, outname), mask)

if __name__ == '__main__':
    data_path = os.path.join(cfg.DATA.IMAGE_PATH, 'test/images/')
    demo(args, data_path)

# python prediction.py --model FCN_8 --backbone vgg16 --dataset potsdam
# python prediction.py --model FCN_Mul --backbone vgg --dataset potsdam
# python prediction.py --model Fuse_FCN --backbone vgg --dataset potsdam
# python prediction.py --model Fuse_all --backbone vgg --dataset potsdam
