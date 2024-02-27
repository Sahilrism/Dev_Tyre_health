import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.evaluation.coco_evaluation import COCOEvaluator
import os,cv2,shutil,glob
from PIL import Image
import time,json
import operator
import numpy as np

from app.electricmeter_number import meter_number_prediction
from app.gas_meter_read import new_ocr

# from azure.cognitiveservices.vision.computervision import ComputerVisionClient
# from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
# from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
# from msrest.authentication import CognitiveServicesCredentials


def configuration_model_meter():
    cfg = get_cfg()
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"))
    # cfg.merge_from_file(
    # "./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# )
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.DEVICE = 'cuda'

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8# set a custom testing    
    cfg.MODEL.WEIGHTS = 'app/meter_crop.pth'
    predictor = DefaultPredictor(cfg)
    return predictor

def meter_cropping(imm,claimid):
    clss=["obj","electric","gas"]
    name=imm.split("/")[-1]
    im=cv2.imread(imm)
    output=predict_display_crop(im)
   # print(output)
    pred_box = output["instances"].pred_boxes.tensor.cpu().numpy()
    pred_class = output["instances"].pred_classes.cpu().numpy()
   # pred_score = output["instances"].scores.cpu().numpy()
    font = cv2.FONT_HERSHEY_SIMPLEX    # org
    org = (50, 50)   
    # fontScale
    fontScale = 0.5   
    # Blue color in BGR
    color = (255, 0, 0)    
    # Line thickness of 2 px
    thickness = 1

    for i in range(len(pred_box)):
        if pred_class[i]==1:
            # cv2.rectangle(im, (pred_box[i][0], pred_box[i][1]), (pred_box[i][2], pred_box[i][3]), (0,255,255), 2)
            # a1=pred_box[i][0]+10
            # a2=pred_box[i][1]-5
            # org=(int(a1),int(a2))
            # image = cv2.putText(im,clss[pred_class[i]] , org, font, 
            #                 fontScale, color, thickness, cv2.LINE_AA)

            cropped_image = im[int(pred_box[i][1]):int(pred_box[i][3]), int(pred_box[i][0]):int(pred_box[i][2])]
            cv2.imwrite(claimid+"/meter_cropped.jpg",cropped_image)

m_start=time.time()

predict_display_crop = configuration_model_meter() 

model_time=time.time()-m_start

def number_extractor(claimid):
    return_code=""
    current_read=[]
    
    # result_read="0"
    sort_dict={}
    js_path = claimid + "/output.json"
    with open(js_path, "r")as fout:
        sort_dict = json.load(fout)
        
    print("sort_dict >>> ",sort_dict)
    print("numbers detected >>> ",list(sort_dict.values()))
    result_read = ""
    
    try:
        coordlist = list(sort_dict.keys())
        print("coordlist >> ",coordlist)
        res = [eval(i) for i in coordlist]

        res.sort()
        print("sorted_res >> ",res)
        
        ###### removing outlier using custom threshold
        filtered_list = []
        threshold = 130.0  # Adjust as needed
        for i in range(len(res)):
            if i!=(len(res)-1):
                if abs(res[i]-res[i+1])<=threshold:
                    filtered_list.append(res[i])
            elif i==(len(res)-1):
                if abs(res[-1]-res[-2])<=threshold:
                    filtered_list.append(res[i])
                    
        print("filtered_list >> ",filtered_list)
        
        for i in filtered_list:
            result_read+=sort_dict[str(i)]
            return_code = 200
        result_read+=" kW/h"
    except Exception as e:
        print("ERRRORRRRRR occured in Position Combining")
        print("Error is",e)
        return_code=241
    
    #####################################
    
    # sort_list=[]
    # if len(sort_dict)>0:
    #     print(sort_dict)
    #     sorted_d = sorted(sort_dict.items(), key=operator.itemgetter(0))
    #     for i in range(len(sorted_d)):
    #         sort_list.append(sorted_d[i][1])

    # print(sort_list)

    # try:
    #     sort_list2=[]
    #     for i in sort_list:
    #         print(type(i))
    #         sort_list2.append(str(i))

    #     join_sort=(''.join(sort_list2))

    #     read_init=join_sort[0]
    #     join_sort=list(join_sort)
    #     read_remain_=[]
    #     if len(join_sort)>1:
    #         read_remain=join_sort[1:]
    #         read_remain_=read_remain
    #     remain_reading=(''.join(join_sort))

    #     res_comb=remain_reading
    #     result_read=res_comb
    #     return_code = 200
    # except Exception as e:
    #     print("ERRRORRRRRR occured in Position Combining")
    #     print("Error is",e)
    #     return_code=241
  
    return result_read ,return_code                                                                           


def meter_read_extract(imm,claimid):  
    
    meter_cropping(imm,claimid)
    
    try:
        aiout = ""
        ai_comb_read = ""
        return_code = 241
        try:
            input_file_path=claimid+"/meter_cropped.jpg"
            meter_number_prediction(input_file_path,claimid)
            ai_comb_read,return_code=number_extractor(claimid)
        except Exception as e:
            print(e)
        
        import torch
        torch.cuda.empty_cache()
        print("reading >> ",ai_comb_read)
        if (ai_comb_read==" kW/h") or (ai_comb_read==""):
            try:
                try:
                    text,textlist = new_ocr(input_file_path)
                except Exception as e:
                    text,textlist = new_ocr(claimid +"/image.jpg")
                print("text_list>>>",textlist)
                print("text>>>",text)
                filteredlist = []
                for i in textlist:
                    iflag = False
                    try:
                        k = float(i)
                        iflag = True
                    except Exception as e:
                        pass
                    if iflag:
                        # if len(i)<=8:
                        filteredlist.append(i)
                print("filetered_list >> ",filteredlist)
                lendef = 0
                if len(filteredlist)>=0:
                    for j in filteredlist:
                        if len(str(j))>lendef:
                            aiout = str(j)
                            lendef = len(str(j))
                # if len(aiout)>=5:
                #     aiout = aiout[:5] + " kW/h"
                # else:
                aiout = aiout + " kW/h"
                return_code = 200
            except Exception as e:
                print("Exception occured>>",e)
                return_code = 241
            ai_comb_read = aiout
            if (ai_comb_read==" kW/h") or (ai_comb_read==""):
                ai_comb_read = ""
                return_code = 241
        return ai_comb_read,return_code

    except Exception as e:
        print(e)
        print("EXCEPTION PLESE CHECK")
        return_code=241

        return "",241
