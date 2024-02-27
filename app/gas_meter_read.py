import cv2,os,shutil
import numpy as np
from PIL import Image
import re,glob
from datetime import datetime
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
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
import time
import operator

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
        if pred_class[i]==2:
            # cv2.rectangle(im, (pred_box[i][0], pred_box[i][1]), (pred_box[i][2], pred_box[i][3]), (0,255,255), 2)
            # a1=pred_box[i][0]+10
            # a2=pred_box[i][1]-5
            # org=(int(a1),int(a2))
            # image = cv2.putText(im,clss[pred_class[i]] , org, font, 
            #                 fontScale, color, thickness, cv2.LINE_AA)

            cropped_image = im[int(pred_box[i][1])-10:int(pred_box[i][3])+10, int(pred_box[i][0]):int(pred_box[i][2])]
            cv2.imwrite(claimid+"/meter_cropped.jpg",cropped_image)
predict_display_crop = configuration_model_meter() 

def new_ocr(path):
    ext_text=[]
    # print("NEW OCR IS RUNNING")
    read_image_path=path
    read_image = open(read_image_path, "rb") #
    subscription_key= "58f8c4d84e564746b494be961f8f8b07" #MALBH512LMM037551
    endpoint= "https://vcocrapi.cognitiveservices.azure.com/" #MAT607146CWH35007
    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
    # Call API with image and raw response (allows you to get the operation location)
    read_response = computervision_client.read_in_stream(read_image, raw=True)
    # Get the operation location (URL with ID as last appendage)
    read_operation_location = read_response.headers["Operation-Location"]
    # Take the ID off and use to get results
    operation_id = read_operation_location.split("/")[-1]
    # Call the "GET" API and wait for the retrieval of the results
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status.lower () not in ['notstarted', 'running']:
            break
       # time.sleep(5)
    # Print results, line by line
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                string1 = line.text
                # print(string1)
                line = string1
                # line = re.sub("[^a-zA-Z0-9]", "", string1)
                # print(line.text)
                tx=line.replace(" ","")
                tx = tx.replace("m3", "")
                tx = tx.replace("m³", "")
                # tx = tx.replace("&", "")
                # tx = tx.replace(":", "")
                # tx = tx.replace(";", "")
                # tx = tx.replace(".", "") 
                # tx = tx.replace("(", "")
                # tx = tx.replace("_", "")
                # tx = tx.replace("/", "")
                # tx = tx.replace("#", "")
                # tx = tx.replace("+", "")
                # tx = tx.replace("%", "")
                # tx = tx.replace("$", "")
                # tx = tx.replace("~", "")
                # tx = tx.replace("IND", "")
                # tx = tx.replace("AND", "")
                # tx = tx.replace(")", "")
                ext_text.append(tx)

    combine_txt = [''.join(ext_text)]
    return combine_txt[0],ext_text


def gas_meter_reading_extract(image_path,claimid):
    
    meter_cropping(image_path, claimid)
    aiout = ""
    cropped_image = claimid+"/meter_cropped.jpg"
    try:    
        text,textlist = new_ocr(cropped_image)
        print("text_list>>>",textlist)
        print("text>>>",text)
        filteredlist = []
        for i in textlist:
            iflag = False
            try:
                k = int(i)
                iflag = True
            except Exception as e:
                pass
            if iflag:
                if len(i)<=8:
                    filteredlist.append(i)
        print("filetered_list >> ",filteredlist)
        lendef = 0
        if len(filteredlist)>=0:
            for j in filteredlist:
                if len(j)>lendef:
                    aiout = str(j)
                    lendef = len(j)
        if len(aiout)>=5:
            aiout = aiout[:5] + " m³"
        else:
            aiout = aiout + " m³"
        return_code = 200
    except Exception as e:
        print("Exception occured>>",e)
        return_code = 241

    return str(aiout),return_code
