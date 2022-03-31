import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='D2Det inference demo')
    parser.add_argument('--config', default='./configs/d2det/Cascade_D2Det_detection_r101_fpn_1x.py', help='test config file path')
    parser.add_argument('--checkpoint', default='./checkpoints/Cascade_D2Det_latest.pth', help='checkpoint file')
    parser.add_argument('--img_file', default='/data/caokun/PycharmProjects/D2Det-mmdet2.1-master/demo/test7.png', help='img path')
    parser.add_argument('--out', default='/data/caokun/PycharmProjects/D2Det-mmdet2.1-master/demo/test_7(3).png', help='output result path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    # test a single image
    result = inference_detector(model, args.img_file)
    # show the results
    show_result_pyplot(model, args.img_file, result, out_file=args.out)


if __name__ == '__main__':
    main()
