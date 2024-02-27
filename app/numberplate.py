import cv2,os,shutil
import numpy as np
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import re
from datetime import datetime

#from app.regex_num import number_regerx
from app.regex_num import number_regerx



def write_txt(claim_id,text):
    with open(claim_id+'/txt_logs.txt', 'a') as f:
        f.write(str(text))


def image_number_plate_axis(path,claim_id):

    return_flag="NO"
    NumMissingWeight = "app/yolo-obj_number_plate.weights"
    NumMissingConfig = "app/yolo-obj_number_plate.cfg"
    rpmnet = cv2.dnn.readNet(NumMissingWeight, NumMissingConfig) 
    label=["pvt","Not Pvt"]

            
    image=cv2.imread(path)
        
    np.random.seed(42)
    (H, W) = image.shape[:2]
    height_,width_ = H,W

    # determine only the "ouput" layers name which we need from YOLO
    ln = rpmnet.getLayerNames()
    ln = [ln[i[0] - 1] for i in rpmnet.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward pass of the YOLO object detector, 
    # giving us our bounding boxes and associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    rpmnet.setInput(blob)
    layerOutputs = rpmnet.forward(ln)

    boxes = []
    confidences = []
    classIDs = []
    threshold = 0.7

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            # confidence type=float, default=0.5
            if confidence > threshold:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)
    color2 = (0,255, 0) 


    # ensure at least one detection exists
    if len(idxs) > 0:
        return_flag="YES"
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            # color = (255,0,0)

            # cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            # text = "{}".format(label[classIDs[i]], confidences[i])
            # cv2.putText(image, text, (x +15, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
            # 0.5, color2, 1)
            w = w + 55
            h = h + 10
            
    
            # cropped_number_image = image[y:y+h, x:x+w]
            cropped_number_image = image[y-10:y+h, 0:width_]
            
          
    
        
        cv2.imwrite(claim_id + "/number_crop.jpg", cropped_number_image)

    crop_path=claim_id + "/number_crop.jpg"

    return return_flag,crop_path

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
                line = re.sub("[^a-zA-Z0-9]", "", string1)
                #print(line.text)
                tx=line.replace(" ","")
                tx = tx.replace("-", "")
                tx = tx.replace("*", "")
                tx = tx.replace("&", "")
                tx = tx.replace(":", "")
                tx = tx.replace(";", "")
                tx = tx.replace(".", "") 
                tx = tx.replace("(", "")
                tx = tx.replace("_", "")
                tx = tx.replace("/", "")
                tx = tx.replace("#", "")
                tx = tx.replace("+", "")
                tx = tx.replace("%", "")
                tx = tx.replace("$", "")
                tx = tx.replace("~", "")
                tx = tx.replace("IND", "")
                tx = tx.replace("AND", "")
                tx = tx.replace(")", "")
                ext_text.append(tx)

    
    
    combine_txt = [''.join(ext_text)]
    return combine_txt[0],ext_text

def number_plate_crop(path,claim_id,API_Request):
    return_code=""
    # if os.path.exists(claim_id):
    #         shutil.rmtree(claim_id)
    # os.mkdir(claim_id + "/")

    ocr_number_extracted=""
    return_number=""
    output_flag,crop_path=image_number_plate_axis(path,claim_id)
    
    no_number="YES"

    if output_flag=="YES":
        service_start=0
        try:
            start=datetime.now()
            service_start=start
            try:
                print("Cropped img OCR called")
                return_text,ext_text=new_ocr(crop_path)
                print("return_text=====>", return_text)
                print("ext_text=====>", ext_text)
                write_txt(claim_id,ext_text)
                ocr_number_extracted=return_text

            except:
                try:
                    imm_g=cv2.imread(crop_path,cv2.IMREAD_GRAYSCALE)
                    cv2.imwrite(claim_id + "/number_crop_gray.jpg", imm_g)
                    return_text,ext_text=new_ocr(claim_id + "/number_crop_gray.jpg")
                    write_txt(claim_id,ext_text)
                    ocr_number_extracted=return_text

                except Exception as e:
                    return_text,ext_text=new_ocr(path)
                    write_txt(claim_id,ext_text)
                    print("return_text=====", return_text)
                    ocr_number_extracted=return_text


            end=datetime.now()
            total_time=(end-start).total_seconds()

            # API_Service_Logs(claim_id,"OCR","numberplate","numberplate",str(API_Request),str(ext_text),"",start,end,total_time)


        except Exception as e:
            print(e)
            end_exception=datetime.now()
            process_time=(end_exception-service_start).total_seconds()
            # API_Service_Logs(claim_id,"OCR","numberplate","numberplate",str(API_Request),"",e,service_start,end_exception,process_time)

    else:
        print("whole img OCR called")
        return_text,ext_text=new_ocr(claim_id + "/image.jpg")
        write_txt(claim_id,ext_text)
        print("return_text=====>", return_text)
        ocr_number_extracted=return_text
    reg_f = "false"
    if ocr_number_extracted!="":
        print("Working for regex")
        return_number=ocr_number_extracted
        ret_num,reg_f=number_regerx(ocr_number_extracted)
        if ret_num!="":
            return_number=ret_num
            return_code=200
                  
        else:
            return_code=201
     
    else:
        return_code=202

    # if os.path.exists(claim_id):
    #         shutil.rmtree(claim_id)
    if reg_f!="true":
        replacenum = return_number[:2].lower()           
        if replacenum == "mh":
            dig = return_number[2:4].lower()
            if dig.isnumeric():
                return_number = return_number
            else:
                if dig[0] == "o":
                    dig = dig.replace("o","0")
                    return_number = replacenum + dig + return_number[4:]
                    return_number = return_number.upper()
                    return_code=200
                    return_number,reg_f=number_regerx(return_number)
        
        elif "mh" in return_number.lower():
            for i in range(len(return_number)):
                if return_number[i:i+2].lower() == "mh":
                    final = return_number[i:]
                    print("This is final:", final)
                    break

            replacenum_new = final[:2].lower()           
            if replacenum_new == "mh":
                dig = final[2:4].lower()
                if dig.isnumeric():
                    return_number = final
                else:
                    if dig[0] == "o":
                        dig = dig.replace("o","0")
                        return_number = replacenum_new + dig + final[4:]
                        return_number = return_number.upper()
                        return_code=200
                        return_number,reg_f=number_regerx(return_number)

    if reg_f!="true":
        return_text,ext_text=new_ocr(claim_id + "/image.jpg")
        write_txt(claim_id,ext_text)
        print("return_text=====>", return_text)
        return_number,reg_f=number_regerx(return_text)
        print("ocr whole number=====>", return_number)
        print("ocr whole flag=====>", reg_f)
        if (return_number!="") and reg_f=="true":
            return_code=200
        else:
            return_code=201

        if reg_f!="true":
            replacenum = return_number[:2].lower()           
            if replacenum == "mh":
                dig = return_number[2:4].lower()
                if dig.isnumeric():
                    return_number = return_number
                else:
                    if dig[0] == "o":
                        dig = dig.replace("o","0")
                        return_number = replacenum + dig + return_number[4:]
                        return_number = return_number.upper()
                        return_code=200
                        return_number,reg_f=number_regerx(return_number)
                
    return return_number,return_code,reg_f