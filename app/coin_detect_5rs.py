import argparse
import time
from pathlib import Path
import numpy as np
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


def plot_one_box_num(lead_id , name , x, img, color=None, label=None, line_thickness=3):
    name = "det_" + name
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        cv2.imwrite(lead_id + "/{}.jpg".format(name) , img)

def coin_detect(save_img=False):
    t11=time.time()
    label_dict={}
    #source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    
    # opt_weights =  "app/coin_cropping.pt"
    # opt_weights =  "app/epoch_3499.pt"
    ############new weight file with upper hand also#############
    opt_weights =  "app/5RS_Detection_5499.pt"

    opt_view_img = False
    opt_save_txt = False
    opt_img_size = 640
    opt_no_trace = True
    weights, view_img, save_txt, imgsz, trace = opt_weights, opt_view_img, opt_save_txt, opt_img_size, not opt_no_trace

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
    
# NumberPlate model for Two-Wheeler Loading
with torch.no_grad():
    model_,device_=coin_detect()
model_NP= model_
device_NP=device_
print("Coin detection Model Loaded")

def coin_prediction_5rs(lead_id):
    hand_location = "Normal"
    resolution_flag = True
    met_f = True
    source = lead_id
    print("claim_id,img_source : ",lead_id , source)
    label_dict={}
    stride=32
    imgsz=640
    save_conf = 0.3
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    print("image loaded >>>>>>>> ",dataset)

    # Get names and colors
    names = model_NP.module.names if hasattr(model_NP, 'module') else model_NP.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    
    device_type="cuda"
    if device_type != 'cpu':
        model_NP(torch.zeros(1, 3, imgsz, imgsz).to(device_NP).type_as(next(model_NP.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    
    half=True
    
    for path, img, im0s, vid_cap in dataset:
        img_name=path.split("/")[-1].split(".")[0]

        img = torch.from_numpy(img).to(device_NP)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        opt_augment=False
        
        if device_NP.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model_NP(img, augment=opt_augment)[0]

        t12=time.time()
        # Inference
        t1 = time_synchronized()
        pred = model_NP(img, augment=opt_augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        opt_conf_thres=0.1
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

        detected_box_dict = dict()
        confprev = 0.30
        #
        coin_x2 = []
        coin_y2 = []
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
            # print('X__'*100)
            # print(det)
            # print("aaaa122",det)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # print("aaaa1",det[:, :4])
                new_det = det[:,:4]
                # print("OKAYOKAYOKAY >>>>>>>",new_det[0][2])

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    save_img=True

                   

                    # Plotting box over actual image
                    # if save_img:  # Add bbox to image
                    #     # label = "10"
                    #     label = f'{names[int(cls)]} {conf:.2f}'
                    #     # plot_one_box_num(lead_id ,img_name ,xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                    #     # plot_one_box_num(lead_id ,img_name , xyxy, im0, label=label, line_thickness=2)
                    
                    label = f'{names[int(cls)]}'
                    conf_ = f'{conf:.2f}'
                    print("conf of detected object >> ",float(conf_),label)
                    # finger_det=[]
                    if float(conf_)>= 0.10:
                        print("confidence > 10")
                        if ("5" not in str(label)) and (float(conf_)>=confprev):
                            print("finger detected here only")
                            x_,y_,w_,h_=int(xyxy[0]) , int(xyxy[1]) , int(xyxy[2] - xyxy[0]) , int(xyxy[3] - xyxy[1])
                            detected_box_dict[label] = [(x_,y_),(w_,h_)]
                            detected_box_dict['finger'] = [(x_,y_),(int(xyxy[2]),int(xyxy[3]))]
                            # print("This is finger box",detected_box_dict['finger'])
                            print("Model Finger Detected Box >>>>>",detected_box_dict)
                            confprev = float(conf_)
                            # finger_det.append(label)
                            # # print("THIS IS LABELLL",finger_det)
                    if "5" in str(label):
                        print('Coin detected here')
                        # cropping detected coin
                        for k in range(len(det)):
                            x,y,w,h=int(xyxy[0]) , int(xyxy[1]) , int(xyxy[2] - xyxy[0]) , int(xyxy[3] - xyxy[1]) 
                            coin_x2 = int(xyxy[2])
                            coin_y2 = int(xyxy[3]) 
                            print("Coin x2 >>>>>>", coin_x2)  
                            print("Coin y2 >>>>>>", coin_y2)                             
                            img_ = im0.astype(np.uint8)
                            # print("x-"*50)
                            print("H W of detected coin >> ",h,w)
                            detected_box_dict[label] = [(x,y),(w,h)]
                            detected_box_dict['10_upper'] = [(x,y),(coin_x2,coin_y2)]
                            print("This is coin box >>>>>",detected_box_dict[label])
                            # print("This is coin box with x2 and y2 >>>>>",detected_box_dict)
                            # if h<=w and met_f:
                            resolution_flag = False
                            met_f = False

                            # crop_img=img_[y-10:y+ h+30, x-10:x + w+20] 
                            crop_img=img_[y-20:y+ h+15, x-5:x + w+10]
                            # print("crop_img >> ",crop_img)
                            filepath = lead_id + "/image.jpg"
                            try:
                                cv2.imwrite(filepath, crop_img)
                                
                            except Exception as e:
                                print(e)
   
                print("coin cropped and saved") 

        print("finger_dict >>>> ",detected_box_dict) 
        
        print("*"*50)
        print("COIN X2 >>>>>>",coin_x2)
        print("COIN Y2 >>>>>>",coin_y2)

        try:
            if "left_finger" in detected_box_dict:
                finger_det="left_finger"
            elif "right_finger" in detected_box_dict:
                finger_det="right_finger"
            else:
                finger_det="No finger detected"

        except Exception as e:
            print("Exception in fingerrr")

        coin_center_x = (detected_box_dict['5'][0][0]+ coin_x2)/2
        coin_center_y = (detected_box_dict['5'][0][1] + coin_y2)/2
        hand_location = "Normal"
        print("*"*50)
        print("Coin Center X >>>>>>",coin_center_x)
        print("Coin Center Y >>>>>>",coin_center_y)

        if 'finger' in detected_box_dict.keys():
            finger_box_center = (detected_box_dict["finger"][0][0] + detected_box_dict["finger"][1][0])/2
            print("FINGER CENTER >>>>>", finger_box_center)
            
            if finger_box_center < coin_center_x and detected_box_dict['finger'][1][0] > coin_center_x:
                hand_location = "Upper"

            elif finger_box_center > coin_center_x and detected_box_dict['finger'][0][0] < coin_center_x:
                hand_location = "Upper"

        elif 'finger' not in detected_box_dict.keys():
            print("FINGER NOT DETECTED ")
            hand_location = "Not Detected"

        print("Hand Location >>>>>",hand_location)

        ##########################################################
        # Save results (image with detections)
        # try:
        #     if save_img:
        #         if dataset.mode == 'image':
        #             cv2.imwrite(save_path, im0)
        #             print(f" The image with the result is saved in: {save_path}")
        #         else:  # 'video' or 'stream'
        #             if vid_path != save_path:  # new video
        #                 vid_path = save_path
        #                 if isinstance(vid_writer, cv2.VideoWriter):
        #                     vid_writer.release()  # release previous video writer
        #                 if vid_cap:  # video
        #                     fps = vid_cap.get(cv2.CAP_PROP_FPS)
        #                     w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #                     h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #                 else:  # stream
        #                     fps, w, h = 30, im0.shape[1], im0.shape[0]
        #                     save_path += '.mp4'
        #                 vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        #             vid_writer.write(im0)
        # except Exception as e:
        #     print(e)
    # ########################################################## 


        return resolution_flag,detected_box_dict,hand_location,finger_det
