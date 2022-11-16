# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
import cv2
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import torchvision_custom
import torch
import dataset.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', default = "/home/xwchi/data/real/dataset2/19/10.png", help='Image file')
    parser.add_argument('--config', default="/home/xwchi/mmdetection/configs/boundary_bq/configuration_trans.py", help='Config file')
    parser.add_argument('--checkpoint', default = "/home/xwchi/QRCNN/result/model_0.pth", help='Checkpoint file')
    parser.add_argument('--out_file', default="/home/xwchi/QRCNN/test.jpg", help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.80, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    x_min, x_max, y_min, y_max = 545, 955, 75, 345
    img = cv2.imread(args.img) # 480,640,3
    # img = img[y_min:y_max,x_min:x_max,:]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    # build the model from a config file and a checkpoint file
    # model = init_detector(args.config, args.checkpoint, device=args.device)
    model = torchvision_custom.models.detection.__dict__[args.model](num_classes=2,
                                                            pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    # test a single image
    input_img = torch.tensor(img).unsqueeze(0).permute(0,3,1,2) / 255.
    model.eval()
    results = model(input_img)
    fig, ax = plt.subplots()
    ax.imshow(img[:, :, [2, 1, 0]])
    for result in results:
        bboxes = result['boxes'].detach().numpy()
        for bbox in bboxes:
            # bbox=[10,10,100,100]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    plt.axis('off')
    plt.savefig(args.out_file)
    # result = inference_detector(model, img)
    # show the results


async def async_main(args):
    # build the model from a config file and a checkpoint file
    # model = init_detector(args.config, args.checkpoint, device=args.device)
    model = torchvision_custom.models.detection.__dict__[args.model](num_classes=2,
                                                            pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
