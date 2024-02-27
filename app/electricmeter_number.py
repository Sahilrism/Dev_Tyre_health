import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def plot_one_box_meter(claim ,name , x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    cv2.imwrite(claim + "/detetctedimage.jpg",img)

        

def detect_meter(save_img=False):
    t11=time.time()
    label_dict={}
    #source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    # opt_weights="app/meter_number_detect.pt"
    opt_weights="app/meter_number_detect.pt"
    opt_source="/root/Downloads/odometer/odo_testing_early_loading/res3"  
    opt_view_img=False
    opt_save_txt=False
    opt_img_size=640
    opt_no_trace=True
    source, weights, view_img, save_txt, imgsz, trace = opt_source, opt_weights, opt_view_img, opt_save_txt, opt_img_size, not opt_no_trace
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")    
   # print(opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace)    
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    # save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    #     ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    opt_project="runs/detect"
    opt_name="exp"
    opt_exist_ok=False
    save_dir = Path(increment_path(Path(opt_project) / opt_name, exist_ok=opt_exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    opt_device=""
    device = select_device(opt_device)
    # device='cuda'
    # half = device.type != 'cpu'  # half precision only supported on CUDA
    half=True

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    opt_img_size=640
    if trace:
        model = TracedModel(model, device, opt_img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    print(stride,imgsz)
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    t22=time.time()-t11
    print("model early loading time taken is:",t22)
    vid_path, vid_writer = None, None
    # if webcam:
    #     view_img = check_imshow()
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    # else:
    return model,device


with torch.no_grad():
    return_model,return_device=detect_meter()

# model= return_model
# device=return_device    
def meter_number_prediction(source,claim_id):
    # using models based on guage type
    model= return_model
    device=return_device
    
    label_dict={}
    stride=32
    imgsz=640
    #source="/root/Downloads/odometer/odo_testing_early_loading/res3"
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    
    device_type="cuda"
    if device_type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    
    half=True
    save_img=True
    
    for path, img, im0s, vid_cap in dataset:
        un_img_name=path.split("/")[-1].split(".")[0]
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        opt_augment=False
        
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt_augment)[0]

        t12=time.time()
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt_augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        opt_conf_thres=0.3
        opt_iou_thres=0.45
        opt_classes=None
        opt_agnostic_nms=False
        pred = non_max_suppression(pred, opt_conf_thres, opt_iou_thres, classes=opt_classes, agnostic=opt_agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        #
        opt_project="runs/detect"
        opt_name="exp"
        opt_exist_ok=False
        save_txt=False
        save_dir = Path(increment_path(Path(opt_project) / opt_name, exist_ok=opt_exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        
        #
        for i, det in enumerate(pred):  # detections per image
            webcam=False
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    cn = float(f'{conf:.2f}')
                    if cn>=0.81:
                        # print(type(cn))
                        print("label , conf >> ",f'{names[int(cls)]} {conf:.2f}')
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        
                        if save_img :  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            s1 = slice(1)
                            s2 = slice(2,8)
                            a1,a2=plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                            # label_dict[label[s1]+"_"+label[s2]+"_"+str(a1[0])]=[a1,a2]
                            
                            label_dict[a1[0]]=names[int(cls)]
                            start = tuple(a1)
                            end = tuple(a2)

                            plot_one_box_meter(claim_id , un_img_name , xyxy, im0, color=None, label=label, line_thickness=1)
        


            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

    #         # Stream results
    #         # if view_img:
    #         #     cv2.imshow(str(p), im0)
    #         #     cv2.waitKey(1)  # 1 millisecond

    #         # Save results (image with detections)
    #         if save_img:
    #             if dataset.mode == 'image':
    #                 cv2.imwrite(save_path, im0)
    #                 print(f" The image with the result is saved in: {save_path}")
    #             else:  # 'video' or 'stream'
    #                 if vid_path != save_path:  # new video
    #                     vid_path = save_path
    #                     if isinstance(vid_writer, cv2.VideoWriter):
    #                         vid_writer.release()  # release previous video writer
    #                     if vid_cap:  # video
    #                         fps = vid_cap.get(cv2.CAP_PROP_FPS)
    #                         w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #                         h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #                     else:  # stream
    #                         fps, w, h = 30, im0.shape[1], im0.shape[0]
    #                         save_path += '.mp4'
    #                     vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    #                 vid_writer.write(im0)

    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    
    import json
    #claim_id="12"
    with open("{}/output.json".format(claim_id), 'w') as f: 
      json.dump(label_dict, f)
    print("Yolov7 Successfully executed")

        