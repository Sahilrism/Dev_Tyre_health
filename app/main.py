from fileinput import filename
import uvicorn
from typing import Optional
from fastapi import FastAPI, File, UploadFile,Form,Header,Request
from PIL import Image
from fastapi import FastAPI,Request,Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image as imm
import cv2 , os, shutil , glob , numpy
import pymysql
import uuid
from datetime import datetime
from io import BytesIO
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from Crypto.Cipher import AES
import base64

# Importing modules required
from app.logs import *
from app.error_Info import error_dict
from app.coin_detect_yolov7 import *
from app.eval_coin import *
from app.coin_calculation import coin_calc
from app.find_defect_outside_file import find_defect_outside
from app.numberplate import number_plate_crop
from app.veh_api_info import api_info
from app.image_api import image_post
from app.coin_mask import draw_img
from app.image_api import new_image
from app.find_defect_outside_file import tiredetect
from app.meter import meter_read_extract
from app.gas_meter_read import gas_meter_reading_extract
from app.coin_detect_5rs import *
from app.eval_coin_5rs import *



def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

app = FastAPI()
AES_SECRET_KEY = 'CETC_AES_256_ENCRYPTION_KEY_2023'# ා 16| 24| 32 characters here
IV = "CETC_AES_256_ENC"

app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["*"]
)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
###################for app settinggg###############

@app.get("/appsetting")
async def ModelLoad(auth_: str = Header(convert_underscores=False)):
    
    BS = len(AES_SECRET_KEY)
    key = AES_SECRET_KEY
    mode = AES.MODE_CBC
    pad = lambda s: s + (BS - len(s) % BS) * chr(BS - len(s) % BS)
    unpad = lambda s: s[0:-ord(s[-1:])]

    def decrypt(text):
        decode = base64.b64decode(text)
        cryptor = AES.new(key.encode("utf8"), mode, IV.encode("utf8"))
        plain_text = cryptor.decrypt(decode)
        return unpad(plain_text)

    # b+str(auth_)
    auth_check=""
    print("*"*100)
    print(auth_)
    print("*"*100)
    try:

        aut=decrypt(auth_)
        auth_check=aut
        print(aut)

    except Exception as e:
        print(e)
        if auth_check!=b'username:password':
            return_result="Bad Authentication!"
        
            return return_result
    

    DataDict = dict()
    # if Type == "setting":
    record = settingchecknew()
    # print("tHIS IS RECCC",record)
    for i in record:
        DataDict[i[0]] = i[1]
    # print(DataDict)
    return DataDict

####################################

@app.get("/apphealth")
async def ModelLoad(auth_: str = Header(convert_underscores=False)):
    
    BS = len(AES_SECRET_KEY)
    key = AES_SECRET_KEY
    mode = AES.MODE_CBC
    pad = lambda s: s + (BS - len(s) % BS) * chr(BS - len(s) % BS)
    unpad = lambda s: s[0:-ord(s[-1:])]

    def decrypt(text):
        decode = base64.b64decode(text)
        cryptor = AES.new(key.encode("utf8"), mode, IV.encode("utf8"))
        plain_text = cryptor.decrypt(decode)
        return unpad(plain_text)

    # b+str(auth_)
    auth_check=""
    print("*"*100)
    print(auth_)
    print("*"*100)
    try:

        aut=decrypt(auth_)
        auth_check=aut
        print(aut)

    except Exception as e:
        print(e)
        if auth_check!=b'username:password':
            return_result="Bad Authentication!"
        
            return return_result
    

    DataDict = dict()
    # if Type == "setting":
    record = settingcheck()
    final=record[0][0]
    final={"Tyreapplicationno":final}
    # for i in record:
    #     DataDict[i[0]] = i[1]
    # # print(DataDict)
    return final

allowed_coin_years = [
    "10 Rs 2010-14",
    "10 Rs 2015-2018",
    "10 Rs 2019-2023",
    "5 Rs 1993-2018",
    "5 Rs 2019-2024"
]

@app.post("/Health_Check")
async def predict_api(request: Request,file: UploadFile = File(...),providerId: str = Form(),claimid: str = Form(),dealerid: str = Form(),servicetype: str = Form(),type: str = Form(),Coin_Year: str = Form(None, enum=allowed_coin_years),reg_number: str = Form(),latitude: Optional[str] = Form(None),longitude: Optional[str] = Form(None),address: Optional[str] = Form(None),specs: Optional[str] = Form(None),extra1: Optional[str] = Form(None),extra2: Optional[str] = Form(None),extra3: Optional[str] = Form(None),auth_: str = Header(convert_underscores=False)):
       
    try:
        sttt = time.time()
        ip = request.client.host
        type=type.lower()
        claimwarranty_actual = claimid

        drop_op=Coin_Year
        print("This is Coin Year",drop_op)
        
        if drop_op =="10 Rs 2010-14":
            extra1 = 1
        elif drop_op =="10 Rs 2015-2018":
            extra1 = 2
        elif drop_op =="10 Rs 2019-2023":
            extra1 = 1
        elif drop_op =="5 Rs 1993-2018":
            extra1 = 4
        elif drop_op =="5 Rs 2019-2024":
            extra1 = 3
        else:
            print("Drop down option not selected")
        
        uid=str(uuid.uuid4())
        claimid=uid+"_"+type
        img_name_post = type+"_"+ uid

        start_universal=datetime.now()
        end_universal=""
        total_difference="" 

        print("@@@###",claimid)
        BS = len(AES_SECRET_KEY)
        key = AES_SECRET_KEY
        mode = AES.MODE_CBC
        pad = lambda s: s + (BS - len(s) % BS) * chr(BS - len(s) % BS)
        unpad = lambda s: s[0:-ord(s[-1:])]

        def decrypt(text):
            decode = base64.b64decode(text)
            cryptor = AES.new(key.encode("utf8"), mode, IV.encode("utf8"))
            plain_text = cryptor.decrypt(decode)
            return unpad(plain_text)

        # b+str(auth_)
        auth_check=""
        print("*"*100)
        print(auth_)
        print("*"*100)

        try:

            aut=decrypt(auth_)
            auth_check=aut
            print(aut)

        except Exception as e:
            print(e)
            if auth_check!=b'username:password':
                return_result="Bad Authentication!"
            
                return return_result
            
        claim_main=dict()
        claim_main[claimid]={}
        AXIS_DICT=claim_main[claimid]
        AXIS_DICT["Tyre_Unqiue_Id"]=claimwarranty_actual
        AXIS_DICT["Type"]=type

        AXIS_DICT["Vehicle_Owner_Name"]= ""
        AXIS_DICT["Vehicle_father_name"] = ""
        AXIS_DICT["Vehicle_Number_Plate"]= ""
        AXIS_DICT["Vehicle_Reg_Date"]= ""
        AXIS_DICT["Vehicle_Manufactured_Date_Year"]= ""
        AXIS_DICT["Vehicle_presentAddress"] = ""
        AXIS_DICT["Vehicle_addressLine"] = ""
        AXIS_DICT["Vehicle_country"] = ""
        AXIS_DICT["Vehicle_state_cd"] = ""
        AXIS_DICT["Vehicle_State"]= ""
        AXIS_DICT["Vehicle_district_name"] = ""
        AXIS_DICT["Vehicle_city_name"] = ""
        AXIS_DICT["Vehicle_pincode"] = ""
        AXIS_DICT["Vehicle_Type"] = ""
        AXIS_DICT["Vehicle_Make"]= ""
        AXIS_DICT["Vehicle_Model"] = ""
        AXIS_DICT["Vehicle_variant"] = ""
        AXIS_DICT["Vehicle_Chassis_Number"]=""
        AXIS_DICT["Vehicle_Engine_Number"]=""
        AXIS_DICT["Vehicle_cd"] = ""
        AXIS_DICT["Vehicle_status"] = ""
        AXIS_DICT["Vehicle_timestamp"] = ""
        AXIS_DICT["Vehicle_ManufacturerName"] = ""
        AXIS_DICT["Vehicle_fuel_type"] = ""
        AXIS_DICT["Vehicle_normsType"] = ""
        AXIS_DICT["Vehicle_bodyType"] = ""
        AXIS_DICT["Vehicle_ownerCount"] = ""
        AXIS_DICT["Vehicle_statusAsOn"] = ""
        AXIS_DICT["Vehicle_regAuthority"] = ""
        AXIS_DICT["Vehicle_rcExpiryDate"] = ""
        AXIS_DICT["Vehicle_TaxUpto"] = ""
        AXIS_DICT["Vehicle_InsuranceCompanyName"] = ""
        AXIS_DICT["Vehicle_InsuranceUpto"] = ""
        AXIS_DICT["Vehicle_InsurancePolicyNumber"] = ""
        AXIS_DICT["Vehicle_rcFinancer"] = ""
        AXIS_DICT["Vehicle_CubicCapacity"] = ""
        AXIS_DICT["Vehicle_unladenWeight"] = ""
        AXIS_DICT["Vehicle_CylindersNo"] = ""
        AXIS_DICT["Vehicle_SeatCapacity"] = ""
        AXIS_DICT["Vehicle_puccNumber"] = ""
        AXIS_DICT["Vehicle_puccUpto"] = ""
        AXIS_DICT["Vehicle_isCommercial"] = ""

        AXIS_DICT["Error_Code"]=""
        AXIS_DICT["Error_Message"]=""
        AXIS_DICT["Tyrewise_AI_Output"]= []

        serviceprams=["claim"]
        servicetype = servicetype.lower()
        
        print("Hello")
        if servicetype not in serviceprams:
            AXIS_DICT["Error_Code"]=208
            AXIS_DICT["Error_Message"]=error_dict["208"]

            start_universal_new=(datetime.now())
            end_universal_new=(datetime.now())
            total_time=(start_universal_new-end_universal_new).total_seconds()
        
            UID = str(claimid)
            Claim_Warranty_Id = claimwarranty_actual
            Dealer_Id = dealerid
            Service_type = servicetype
            Type = type
            Model_OCR = "In-House Model"
            API_Request = ""
            API_Output = str(AXIS_DICT)
            Exception_Occurred = AXIS_DICT["Error_Message"]
            Request_Received = str(start_universal)
            Request_Proccessed = str(end_universal)
            Processing_Time = total_time
            AI_Output = ""
            Image_name = img_name_post + ".jpg"
            Remark = ""
            Remark_Status = ""

            tyreHealth_model_logs(UID, Claim_Warranty_Id, Dealer_Id, Service_type, Type, Model_OCR, API_Request, API_Output, Exception_Occurred, Request_Received, Request_Proccessed, Processing_Time, AI_Output, Image_name, Remark, Remark_Status,latitude,longitude,address,specs,ip,extra1,extra2,extra3)
            return AXIS_DICT
        
        try:
            dealerparams = dealers()
        except Exception as e:
            dealerparams=["0553","3728","0750","9999"]

        if dealerid not in dealerparams:
            AXIS_DICT["Error_Code"]=209
            AXIS_DICT["Error_Message"]=error_dict["209"]
            
            start_universal_new=(datetime.now())
            end_universal_new=(datetime.now())
            total_time=(start_universal_new-end_universal_new).total_seconds()
             
            UID = str(claimid)
            Claim_Warranty_Id = claimwarranty_actual
            Dealer_Id = dealerid
            Service_type = servicetype
            Type = type
            Model_OCR = "In-House Model"
            API_Request = ""
            API_Output = str(AXIS_DICT)
            Exception_Occurred = AXIS_DICT["Error_Message"]
            Request_Received = str(start_universal)
            Request_Proccessed = str(end_universal)
            Processing_Time = total_time
            AI_Output = str(AXIS_DICT)
            Image_name = img_name_post + ".jpg"
            Remark = ""
            Remark_Status = ""

            tyreHealth_model_logs(UID, Claim_Warranty_Id, Dealer_Id, Service_type, Type, Model_OCR, API_Request, API_Output, Exception_Occurred, Request_Received, Request_Proccessed, Processing_Time, AI_Output, Image_name, Remark, Remark_Status,latitude,longitude,address,specs,ip,extra1,extra2,extra3)
            return AXIS_DICT 
        
        API_Request={"providerId":providerId,"claimid":claimid,"dealerid":dealerid,"service_type":servicetype,"type":type,"reg_number":reg_number,"Authorization":auth_}
        
        print(providerId)
        print("*"*75)
        print(API_Request)
        print("*"*75)
        extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png","JPG")
        print("extension >>>>> ",extension)
        image_name = file.filename
        print("image_name === ",image_name)

        
        
        if not extension:
            return "Image must be jpg or png format!"
            
        image = Image.open(file.file)
        # image_new = Image.open(BytesIO(file.file.read()))
        size_kb =len(file.file.read())/1024
        # print("Thssssskbbb",size_kb)
        # print("jjjjjjjjsskbbb",image_new)

        if image.mode in ("RGBA","P"):
            image = image.convert("RGB")
        # image = read_imagefile(await file.read())
        # print(type(image))
    
        pix = numpy.array(image)
        data = imm.fromarray(pix)
        
        
        # For saving image locally
        folder_exists = "False"
        if os.path.exists("uploaded_files/"+claimwarranty_actual+"/"):
            # shutil.rmtree(claimwarranty_actual)
            folder_exists = "True"
            print("*"*100)
            print("It exists and is removed")    
        if folder_exists == "False":
            os.mkdir("uploaded_files/"+claimwarranty_actual+"/")
        file_name = file.filename.split(".")[0]
        # data.save(f"uploaded_files/"+str(claimwarranty_actual)+"/"+{file_name}+".jpg")
        
        if type == "numberplate":
            if os.path.exists(f"uploaded_files/{claimwarranty_actual}/numberplate_1.jpg"):
                paths = f"uploaded_files/{claimwarranty_actual}/"
                dirs = os.listdir(paths)
                # print("[:-:]"*20)
                # print(dirs)
                numberplates = []
                for i in dirs:
                    if "numberplate" in i:
                        numberplates.append(i)

                sort_numberplates = sorted(numberplates)
                print("NUMBER PLATES >>>",sort_numberplates)
                fname_increment = (int(sort_numberplates[-1].split(".")[0][-1]) + 1)
                print("FNAME >>>>>",fname_increment)               
                data.save("uploaded_files/{}/numberplate_{}.jpg".format(claimwarranty_actual,fname_increment))

            else:
                count = 1
                data.save("uploaded_files/{}/numberplate_{}.jpg".format(claimwarranty_actual,count))
                
        elif type == "defect-outside":
            if os.path.exists(f"uploaded_files/{claimwarranty_actual}/defect-outside_1.jpg"):
                paths = f"uploaded_files/{claimwarranty_actual}/"
                dirs = os.listdir(paths)
                # print("[:-:]"*20)
                # print(dirs)
                defects = []
                for i in dirs:
                    if "defect-outside" in i:
                        defects.append(i)

                sort_defects = sorted(defects)
                print("DEFECTS >>>",sort_defects)
                fname_increment = (int(sort_defects[-1].split(".")[0][-1]) + 1)
                print("FNAME >>>>>",fname_increment)               
                data.save("uploaded_files/{}/defect-outside_{}.jpg".format(claimwarranty_actual,fname_increment))

            else:
                count = 1
                data.save("uploaded_files/{}/defect-outside_{}.jpg".format(claimwarranty_actual,count))
            
        elif type == "coin":
            if os.path.exists(f"uploaded_files/{claimwarranty_actual}/coin_1.jpg"):
                paths = f"uploaded_files/{claimwarranty_actual}/"
                dirs = os.listdir(paths)
                # print("[:-:]"*20)
                # print(dirs)
                coins = []
                for i in dirs:
                    if "coin" in i:
                        coins.append(i)

                sort_coins = sorted(coins)
                print("DEFECTS >>>",sort_coins)
                fname_increment = (int(sort_coins[-1].split(".")[0][-1]) + 1)
                print("FNAME >>>>>",fname_increment)               
                data.save("uploaded_files/{}/coin_{}.jpg".format(claimwarranty_actual,fname_increment))

            else:
                count = 1
                data.save("uploaded_files/{}/coin_{}.jpg".format(claimwarranty_actual,count))
        
        
        # print(data.shape)
        if os.path.exists(claimid):
            shutil.rmtree(claimid)
        
        os.mkdir(claimid + "/")

        
        image_name = image_name.replace(" ","")
        
        data.save(str(claimid)+'/image.jpg')
        # data.save(''+str(claimid)+'/image.jpg')
        app_im=cv2.imread(str(claimid)+'/image.jpg')


        input_file_path=claimid+"/image.jpg"
        im_ = cv2.imread(input_file_path)
        img_shape = im_.shape
        print("image shape >>>>> ",img_shape)
        height,width = img_shape[0],img_shape[1]

        # file_path = input_file_path
        # file_size = os.path.getsize(file_path)

        # print("File size in Bytes: " + str(file_size))
        # print("File size in Kilobytes: " + str(file_size/1024))
        # print("File size in Megabytes: " + str(file_size/1024**2))
        # print("File size in Gigabytes: " + str(file_size/1024**3))


        # print("heighttt...",height)
        # print("Width...",width)
        # return 0,0,0
        # if height<width:
        #     print("image is Horizantal..........")
        #     output_file_path = input_file_path

        # else:
        #     print("tHIS IOS ELSS")
        #     rotate_degrees = -90
        #     img = Image.open(input_file_path)
        #     img2 = img.rotate(rotate_degrees, expand=True)
        #     output_file_path = claimid+"/coin_crop_rotated.jpg"
        #     img2.save(output_file_path)
        
        typeparams=["gauge","numberplate","defect-outside","coin","meter","gas_meter"]
        
        if type not in typeparams:
            AXIS_DICT["Error_Code"]=210
            AXIS_DICT["Error_Message"]=error_dict["210"]
            start_universal_new=(datetime.now())
            end_universal_new=(datetime.now())
            total_time=(start_universal_new-end_universal_new).total_seconds()
         
            UID = str(claimid)
            Claim_Warranty_Id = claimwarranty_actual
            # Dealer_Id = API_Request["dealerid"]
            # Service_type = API_Request["service_type"]
            Dealer_Id = dealerid
            Service_type = servicetype
            Type = type
            Model_OCR = "In-House Model"
            # API_Request = str(API_Request)
            API_Request = ""
            API_Output = str(AXIS_DICT)
            Exception_Occurred = AXIS_DICT["Error_Message"]
            Request_Received = str(start_universal)
            Request_Proccessed = str(end_universal)
            Processing_Time = total_time
            AI_Output = ""
            Image_name = img_name_post + ".jpg"
            Remark = ""
            Remark_Status = ""

            tyreHealth_model_logs(UID, Claim_Warranty_Id, Dealer_Id, Service_type, Type, Model_OCR, API_Request, API_Output, Exception_Occurred, Request_Received, Request_Proccessed, Processing_Time, AI_Output, Image_name, Remark, Remark_Status,latitude,longitude,address,specs,ip,extra1,extra2,extra3)
            return AXIS_DICT
        
        
        claim_valid = ["numberplate","defect-outside","gauge","coin","meter","gas_meter"]
        
        if servicetype=="claim":
            if type not in claim_valid:
                AXIS_DICT["Error_Code"]=245
                AXIS_DICT["Error_Message"]=error_dict["245"]
                
                start_universal_new=(datetime.now())
                end_universal_new=(datetime.now())
                total_time=(start_universal_new-end_universal_new).total_seconds()
                
                UID = str(claimid)
                Claim_Warranty_Id = claimwarranty_actual
                Dealer_Id = dealerid
                Service_type = servicetype
                Type = type
                Model_OCR = "In-House Model"
                API_Request = ""
                API_Output = str(AXIS_DICT)
                Exception_Occurred = AXIS_DICT["Error_Message"]
                Request_Received = str(start_universal)
                Request_Proccessed = str(end_universal)
                Processing_Time = total_time
                AI_Output = ""
                Image_name = img_name_post + ".jpg"
                Remark = ""
                Remark_Status = ""
                return AXIS_DICT

    #############Calling to post image####################
        # print("JJJJJJJJJJ1111",img_name_post)
        # print("JJJJJJJJJJ2222",claimid)
        # print("JJJJJJJJJJ3333",API_Request["reg_number"])
        # print("JJJJJJJJJJ4444",providerId)
        # print("JJJJJJJJJJ5555",claimwarranty_actual)
        # try:
        #     start = time.time()
        #     image_post(img_name_post, claimid , API_Request["reg_number"] ,providerId ,claimwarranty_actual)
        #     # image_post(img_name_post, claimid , API_Request["reg_number"] ,providerId ,claimwarranty_actual)
        #     # print("Image sent to External/Frontend API")
        # except Exception as e:
        #     try:         
        #         print("Exception occured in posting data in first attempt")
        #         print("second attempt called")
        #         image_post(img_name_post, claimid , API_Request["reg_number"] ,providerId ,claimwarranty_actual)
        #     except Exception as e:
        #         print("Exception occured in posting data in second attempt")
        # end = time.time() - start
        # print("Time taken for Posting image",end) 

    ############################################################
        def get_coin_size_by_id(data_tuple, target_id):
                for inner_tuple in data_tuple:
                    if inner_tuple[0] == target_id:
                        return inner_tuple
                return None  # ID not mentioned in db
            
        if type=="coin":
            start = time.time()           
            claim_id = claimid
            reading = ""
            remark = ""
            resol_f = False
            mask= ""
            coin_det_condition = "False"

            data = coin_size()
            # print(data)
            target_id = int(extra1)
            
            result = get_coin_size_by_id(data, target_id)
            ID = result[0] 
            coin_name = result[1]
            coin_size_full = result[2]
            coin_mint_year = result[3]
            print(f"ID passed for coin size to db is {extra1}")

            if (coin_name=="10_rs") and (coin_mint_year=="2019-2024"):
                coin_det_condition = "True"
                model_id = 1    
                # To crop coin
                try:
                    resol_f,fingerdict,hand_loc,finger_det = coin_prediction(claim_id)
                    print("*"*50)
                    print("Hand Location detected by model",hand_loc,finger_det)
                    print("*"*50)
                except Exception as e:
                    print(e)
                if not(resol_f):
                    # To detect and save masked array into json
                    try:
                        coin_mask_func(claim_id)
                    except Exception as e:
                        print(e)
            
            elif (coin_name=="10_rs") and (coin_mint_year=="2015-2018"): 
                coin_det_condition = "True"    
                # To crop coin
                try:
                    resol_f,fingerdict,hand_loc,finger_det = coin_prediction(claim_id)
                    print("*"*50)
                    print("Hand Location detected by model",hand_loc,finger_det)
                    print("*"*50)
                except Exception as e:
                    print(e)
                if not(resol_f):
                    # To detect and save masked array into json
                    try:
                        coin_mask_func(claim_id)
                    except Exception as e:
                        print(e)
                    
            elif (coin_name=="5_rs") and (coin_mint_year=="1993-2018"): 
                coin_det_condition = "True"    
                # To crop coin
                try:
                    resol_f,fingerdict,hand_loc,finger_det = coin_prediction_5rs(claim_id)
                    print("*"*50)
                    print("Hand Location detected by model",hand_loc,finger_det)
                    print("*"*50)
                except Exception as e:
                    print(e)
                if not(resol_f):
                    # To detect and save masked array into json
                    try:
                        coin_mask_func_5rs(claim_id)
                    except Exception as e:
                        print(e)

            elif (coin_name=="5_rs") and (coin_mint_year=="2019-2024"): 
                coin_det_condition = "True"    
                # To crop coin
                try:
                    resol_f,fingerdict,hand_loc,finger_det = coin_prediction_5rs(claim_id)
                    print("*"*50)
                    print("Hand Location detected by model >>>>>>>>>",hand_loc,finger_det)
                    print("*"*50)
                except Exception as e:
                    print(e)
                if not(resol_f):
                    # To detect and save masked array into json
                    try:
                        coin_mask_func_5rs(claim_id)
                    except Exception as e:
                        print(e)

            else:
                print("Provide proper ID")

            if coin_det_condition=="True":    
                try:
                    reading,logdata = coin_calc(claim_id,fingerdict,hand_loc,finger_det,coin_size_full)
                    print("Reading>>",reading)
                    # print("THSSSSSSSS",logdata)

                    apiResponse_json = json.dumps(logdata)
                    tyre_logs(claimwarranty_actual,"",apiResponse_json)

                except Exception as e:
                    print(e)
            else:
                reading = ""
            
            
                
            if ("None" in str(reading)) or (str(reading)==""):
                AXIS_DICT["Error_Code"]=243
                AXIS_DICT["Error_Message"]=error_dict["243"]
            else:
                defect_res = get_out_defect(str(claimwarranty_actual),"defect-outside")
                print("Defect_Result>>",defect_res)
                # print("aasswwa")
                if reading > 0:
                    print("GB >>> 1")
                    if reading <= 3:
                        # print("aaqwsswwa")
                        AXIS_DICT["Error_Code"]=214
                        AXIS_DICT["Error_Message"]=error_dict["214"]
                        remark = "Poor"
                        
                    elif reading > 3 and reading <=5 and defect_res == "Good" :
                        print("GB >>> 2")
                        AXIS_DICT["Error_Code"]=215
                        AXIS_DICT["Error_Message"]=error_dict["215"]
                        remark = "Good"
                        
                    elif reading  > 3 and reading <=5 and defect_res == "Bad" :
                        print("GB >>> 3")
                        AXIS_DICT["Error_Code"]=216
                        AXIS_DICT["Error_Message"]=error_dict["216"]
                        remark = "Poor"

                    elif reading > 5 and reading <=7 and defect_res == "Good" :
                        print("GB >>> 4")
                        AXIS_DICT["Error_Code"]=215
                        AXIS_DICT["Error_Message"]=error_dict["215"]
                        remark = "Great"
                        
                    elif reading  > 5 and reading <=7 and defect_res == "Bad" :
                        print("GB >>> 5")
                        AXIS_DICT["Error_Code"]=216
                        AXIS_DICT["Error_Message"]=error_dict["216"]
                        remark = "Good"
                        
                    elif reading  > 7 and defect_res == "Good" :
                        print("GB >>> 6")
                        AXIS_DICT["Error_Code"]=217
                        AXIS_DICT["Error_Message"]=error_dict["217"]
                        remark = "Excellent"
                        
                    elif reading  > 7 and defect_res == "Bad" :
                        print("GB >>> 7")
                        AXIS_DICT["Error_Code"]=218
                        AXIS_DICT["Error_Message"]=error_dict["218"]
                        remark = "Great"

                    elif reading  > 3 and defect_res =="":
                        print("GB >>> 8")
                        AXIS_DICT["Error_Code"]=219
                        AXIS_DICT["Error_Message"]=error_dict["219"]+", coin_Reading:"+str(reading)+" mm"
                        remark = "sidewall image missing"
                        
                else:
                    print("GB >>> 9")
                    AXIS_DICT["Error_Code"]=243
                    AXIS_DICT["Error_Message"]=error_dict["243"]
            # print("SSSSS",resol_f)       
            if resol_f:
                print("aasswwabbbbbbbb")
                AXIS_DICT["Error_Code"]=260
                AXIS_DICT["Error_Message"]=error_dict["260"]

            if ("None" in str(reading)) or (str(reading)==""):
                # print("aasswwaeeeeeeehh")
                sr_one_dict={"Tyre_No": 1,
                    "Tyre_Serial":"",
                    "Photo_Name":"tyre.jpg",
                    "Remark":"",
                    "Coin_Depth_Result":"",
                    "Gauge_Result": "",
                    "Defect_Result" :""               
                    }
            else:
                print("Reading GOT")
                sr_one_dict={"Tyre_No": 1,
                    "Tyre_Serial":"",
                    "Photo_Name":"tyre.jpg",
                    "Remark":remark,
                    "Coin_Depth_Result":str(reading) + " mm",
                    "Gauge_Result": "",
                    "Defect_Result" :""               
                    }
            AXIS_DICT["Tyrewise_AI_Output"].append(sr_one_dict)
            AXIS_DICT["Type"] = type
            end = time.time() - start
            print("Time taken for coin process",end) 

            try:
                print("aasswwammmmmmmmm")
                end_defect=datetime.now()
                end_universal=end_defect
                total_time=(end_defect-start_universal).total_seconds()
                total_difference=total_time
                
                UID = str(claimid)
                Claim_Warranty_Id = claimwarranty_actual
                Dealer_Id = dealerid
                Service_type = servicetype
                Type = type
                Model_OCR = "In-House Model"
                API_Request = str(API_Request)
                API_Output = str(AXIS_DICT)
                Exception_Occurred = AXIS_DICT["Error_Message"]
                Request_Received = str(start_universal)
                Request_Proccessed = str(end_universal)
                Processing_Time = total_time
                AI_Output = str(reading) + " mm"
                Image_name = img_name_post + ".jpg"
                Remark = ""
                Remark_Status = ""

                tyreHealth_model_logs(UID, Claim_Warranty_Id, Dealer_Id, Service_type, Type, Model_OCR, API_Request, API_Output, Exception_Occurred, Request_Received, Request_Proccessed, Processing_Time, AI_Output, Image_name, Remark, Remark_Status,latitude,longitude,address,specs,ip,extra1,extra2,extra3)
                
                photo_details(Claim_Warranty_Id,"3",Image_name,Service_type,"AI",latitude,longitude,address,"AI",AI_Output,"AI", API_Output, Remark,AXIS_DICT["Error_Code"], Exception_Occurred)
                
            except Exception as e:
                print("aasswwaggggggggggggggg")
                print(e)
                AXIS_DICT["Type"] = type
                
                end_defect=datetime.now()
                end_universal=end_defect
                total_time=(end_defect-start_universal).total_seconds()
                total_difference=total_time
                
                UID = str(claimid)
                Claim_Warranty_Id = claimwarranty_actual
                Dealer_Id = dealerid
                Service_type = servicetype
                Type = type
                Model_OCR = "In-House Model"
                API_Request = str(API_Request)
                API_Output = ""
                Exception_Occurred = AXIS_DICT["Error_Message"]
                Request_Received = str(start_universal)
                Request_Proccessed = str(end_universal)
                Processing_Time = total_time
                AI_Output = ""
                Image_name = img_name_post + ".jpg"
                Remark = ""
                Remark_Status = ""

                tyreHealth_model_logs(UID, Claim_Warranty_Id, Dealer_Id, Service_type, Type, Model_OCR, API_Request, API_Output, Exception_Occurred, Request_Received, Request_Proccessed, Processing_Time, AI_Output, Image_name, Remark, Remark_Status,latitude,longitude,address,specs,ip,extra1,extra2,extra3)

                photo_details(Claim_Warranty_Id,"3",Image_name,Service_type,"AI",latitude,longitude,address,"AI",AI_Output,"AI", API_Output, Remark,AXIS_DICT["Error_Code"], Exception_Occurred)
                
            
            return AXIS_DICT

        if type=="numberplate" and reg_number=="NA":
            
            try:
                start = time.time() 
                vehicle_number,return_code,flag_regex=number_plate_crop(str(claimid)+"/image.jpg",str(claimid),API_Request)
                print("flag_regex : ",flag_regex)
                print("Vehicle_number : ",vehicle_number)
                print("return code : ",return_code)

                if return_code==201:
                    AXIS_DICT["Error_Code"]=201
                    AXIS_DICT["Error_Message"]=error_dict["201"]

                if return_code==202:
                    AXIS_DICT["Error_Code"]=202
                    AXIS_DICT["Error_Message"]=error_dict["202"]

                if return_code==200:
                    AXIS_DICT["Error_Code"]=200
                    AXIS_DICT["Error_Message"]=error_dict["200"]

                prediction = "Vehicle Number is:"+str(vehicle_number)
                prediction_=prediction
                AXIS_DICT["Vehicle_Number_Plate"]=str(vehicle_number)

                sr_one_dict={"Tyre_No": 1,
                    "Tyre_Serial":"",
                    "Photo_Name":"tyre.jpg",
                    "Remark":"",
                    "Coin_Depth_Result":"",
                    "Gauge_Result":"",
                    "Defect_Result" :""
                    }

                AXIS_DICT["Tyrewise_AI_Output"].append(sr_one_dict)
                end = time.time() - start
                print("Time taken for NP process",end) 

                end_numberplate=datetime.now()
                end_universal=end_numberplate
                total_time=(end_numberplate-start_universal).total_seconds()
                total_difference=total_time
                
                UID = str(claimid)
                Claim_Warranty_Id = claimwarranty_actual
                Dealer_Id = dealerid
                Service_type = servicetype
                Type = "numberplate"
                Model_OCR = "In-House Model+OCR"
                API_Request = str(API_Request)
                API_Output = str(AXIS_DICT)
                Exception_Occurred = AXIS_DICT["Error_Message"]
                Request_Received = str(start_universal)
                Request_Proccessed = str(end_universal)
                Processing_Time = total_time
                AI_Output = str(vehicle_number)
                Image_name = img_name_post + ".jpg"
                Remark = ""
                Remark_Status = ""

                tyreHealth_model_logs(UID, Claim_Warranty_Id, Dealer_Id, Service_type, Type, Model_OCR, API_Request, API_Output, Exception_Occurred, Request_Received, Request_Proccessed, Processing_Time, AI_Output, Image_name, Remark, Remark_Status,latitude,longitude,address,specs,ip,extra1,extra2,extra3)

                photo_details(Claim_Warranty_Id,"1",Image_name,Service_type,"AI",latitude,longitude,address,"AI",AI_Output,"AI", API_Output, Remark,AXIS_DICT["Error_Code"], Exception_Occurred)

                try:
                    start = time.time() 
                    vehicle_number_info = ""
                    if flag_regex == "true":
                        vehicle_number_info,return_code=api_info(AXIS_DICT["Vehicle_Number_Plate"],claimwarranty_actual,API_Request)

                        # AXIS_DICT["Vehicle_Number_Plate"]=AXIS_DICT["Vehicle_Number_Plate"]
                        # AXIS_DICT["Vehicle_Owner_Name"]= vehicle_number_info["Vehicle_Owner_Name"]
                        # AXIS_DICT["Vehicle_Manufactured_Date_Year"]= vehicle_number_info["Vehicle_Manufactured_Date_Year"]
                        # AXIS_DICT["Vehicle_Reg_Date"]= vehicle_number_info["Vehicle_Reg_Date"]
                        # AXIS_DICT["Vehicle_Make"]=vehicle_number_info["Vehicle_Make"]
                        # AXIS_DICT["Vehicle_Model"]=vehicle_number_info["Vehicle_Model"]
                        # AXIS_DICT["Vehicle_Type"]=vehicle_number_info["Vehicle_Type"]
                        # AXIS_DICT["Vehicle_Chassis_Number"]=vehicle_number_info["Vehicle_Chassis_Number"]
                        # AXIS_DICT["Vehicle_Engine_Number"]=vehicle_number_info["Vehicle_Engine_Number"]
                        # AXIS_DICT["Vehicle_State"]=vehicle_number_info["Vehicle_State"]
                        
                        AXIS_DICT["Vehicle_Owner_Name"]= vehicle_number_info["Vehicle_Owner_Name"]
                        AXIS_DICT["Vehicle_father_name"] = vehicle_number_info["Vehicle_father_name"]
                        AXIS_DICT["Vehicle_Number_Plate"]= vehicle_number_info["Vehicle_Number_Plate"]
                        AXIS_DICT["Vehicle_Reg_Date"]= vehicle_number_info["Vehicle_Reg_Date"]
                        AXIS_DICT["Vehicle_Manufactured_Date_Year"]= vehicle_number_info["Vehicle_Manufactured_Date_Year"]
                        AXIS_DICT["Vehicle_presentAddress"] = vehicle_number_info["Vehicle_presentAddress"]
                        AXIS_DICT["Vehicle_addressLine"] = vehicle_number_info["Vehicle_addressLine"]
                        AXIS_DICT["Vehicle_country"] = vehicle_number_info["Vehicle_country"]
                        AXIS_DICT["Vehicle_state_cd"] = vehicle_number_info["Vehicle_state_cd"]
                        AXIS_DICT["Vehicle_State"]=vehicle_number_info["Vehicle_State"]
                        AXIS_DICT["Vehicle_district_name"] = vehicle_number_info["Vehicle_district_name"]
                        AXIS_DICT["Vehicle_city_name"] = vehicle_number_info["Vehicle_city_name"]
                        AXIS_DICT["Vehicle_pincode"] = vehicle_number_info["Vehicle_pincode"]
                        AXIS_DICT["Vehicle_Type"] = vehicle_number_info["Vehicle_Type"]
                        AXIS_DICT["Vehicle_Make"]=vehicle_number_info["Vehicle_Make"]
                        AXIS_DICT["Vehicle_Model"] = vehicle_number_info["Vehicle_Model"]
                        AXIS_DICT["Vehicle_variant"] = vehicle_number_info["Vehicle_variant"]
                        AXIS_DICT["Vehicle_Chassis_Number"]=vehicle_number_info["Vehicle_Chassis_Number"]
                        AXIS_DICT["Vehicle_Engine_Number"]=vehicle_number_info["Vehicle_Engine_Number"]
                        AXIS_DICT["Vehicle_cd"] = vehicle_number_info["Vehicle_cd"]
                        AXIS_DICT["Vehicle_status"] = vehicle_number_info["Vehicle_status"]
                        AXIS_DICT["Vehicle_timestamp"] = vehicle_number_info["Vehicle_timestamp"]
                        AXIS_DICT["Vehicle_ManufacturerName"] = vehicle_number_info["Vehicle_ManufacturerName"]
                        AXIS_DICT["Vehicle_fuel_type"] = vehicle_number_info["Vehicle_fuel_type"]
                        AXIS_DICT["Vehicle_normsType"] = vehicle_number_info["Vehicle_normsType"]
                        AXIS_DICT["Vehicle_bodyType"] = vehicle_number_info["Vehicle_bodyType"]
                        AXIS_DICT["Vehicle_ownerCount"] = vehicle_number_info["Vehicle_ownerCount"]
                        AXIS_DICT["Vehicle_statusAsOn"] = vehicle_number_info["Vehicle_statusAsOn"]
                        AXIS_DICT["Vehicle_regAuthority"] = vehicle_number_info["Vehicle_regAuthority"]
                        AXIS_DICT["Vehicle_rcExpiryDate"] = vehicle_number_info["Vehicle_rcExpiryDate"]
                        AXIS_DICT["Vehicle_TaxUpto"] = vehicle_number_info["Vehicle_TaxUpto"]
                        AXIS_DICT["Vehicle_InsuranceCompanyName"] = vehicle_number_info["Vehicle_InsuranceCompanyName"]
                        AXIS_DICT["Vehicle_InsuranceUpto"] = vehicle_number_info["Vehicle_InsuranceUpto"]
                        AXIS_DICT["Vehicle_InsurancePolicyNumber"] = vehicle_number_info["Vehicle_InsurancePolicyNumber"]
                        AXIS_DICT["Vehicle_rcFinancer"] = vehicle_number_info["Vehicle_rcFinancer"]
                        AXIS_DICT["Vehicle_CubicCapacity"] = vehicle_number_info["Vehicle_CubicCapacity"]
                        AXIS_DICT["Vehicle_unladenWeight"] = vehicle_number_info["Vehicle_unladenWeight"]
                        AXIS_DICT["Vehicle_CylindersNo"] = vehicle_number_info["Vehicle_CylindersNo"]
                        AXIS_DICT["Vehicle_SeatCapacity"] = vehicle_number_info["Vehicle_SeatCapacity"]
                        AXIS_DICT["Vehicle_puccNumber"] = vehicle_number_info["Vehicle_puccNumber"]
                        AXIS_DICT["Vehicle_puccUpto"] = vehicle_number_info["Vehicle_puccUpto"]
                        AXIS_DICT["Vehicle_isCommercial"] = vehicle_number_info["Vehicle_isCommercial"]

                    
                    prediction = "Vehicle Number is:"+str(vehicle_number_info)
                    prediction_=prediction

                    if AXIS_DICT["Vehicle_Number_Plate"] == None:
                        AXIS_DICT["Error_Code"]=203
                        AXIS_DICT["Error_Message"]=error_dict["203"]
                    
                    if return_code==203:
                        AXIS_DICT["Error_Code"]=203
                        AXIS_DICT["Error_Message"]=error_dict["203"]

                    if AXIS_DICT["Error_Code"]=="":
                        AXIS_DICT["Error_Code"]=200
                        AXIS_DICT["Error_Message"]=error_dict["200"]

                    end = time.time() - start
                    AXIS_DICT["Type"] = "vahaninfo"
                    # AXIS_DICT["Type"] = type

                    end_numberplate_only=datetime.now()
                    total_time=(end_numberplate_only-start_universal).total_seconds()
                    
                    UID = str(claimid)
                    Claim_Warranty_Id = claimwarranty_actual
                    Dealer_Id = dealerid
                    Service_type = servicetype
                    Type = "vahaninfo"
                    Model_OCR = "In-House Model+OCR"
                    API_Request = str(API_Request)
                    API_Output = str(AXIS_DICT)
                    Exception_Occurred = AXIS_DICT["Error_Message"]
                    Request_Received = str(start_universal)
                    Request_Proccessed = str(end_universal)
                    Processing_Time = total_time
                    AI_Output = str(vehicle_number_info)
                    Image_name = img_name_post + ".jpg"
                    Remark = ""
                    Remark_Status = ""

                    vehicle_details(Claim_Warranty_Id,Service_type,AXIS_DICT["Vehicle_Number_Plate"],AXIS_DICT["Vehicle_Manufactured_Date_Year"],AXIS_DICT["Vehicle_Reg_Date"],AXIS_DICT["Vehicle_Owner_Name"],AXIS_DICT["Vehicle_Model"],AXIS_DICT["Vehicle_Make"],AXIS_DICT["Vehicle_Type"],AXIS_DICT["Vehicle_Chassis_Number"],AXIS_DICT["Vehicle_Engine_Number"],AXIS_DICT["Vehicle_State"],"AI","","")

                    tyreHealth_model_logs(UID, Claim_Warranty_Id, Dealer_Id, Service_type, Type, Model_OCR, API_Request, API_Output, Exception_Occurred, Request_Received, Request_Proccessed, Processing_Time, AI_Output, Image_name, Remark, Remark_Status,latitude,longitude,address,specs,ip,extra1,extra2,extra3)

                    # photo_details(Claim_Warranty_Id,"1",Image_name,Service_type,"AI",latitude,longitude,address,"AI",str(vehicle_number),"AI", API_Output, Remark,AXIS_DICT["Error_Code"], Exception_Occurred)
                    
                    # vehicle_details(Claim_Warranty_Id,Service_type,AXIS_DICT["Vehicle_Number_Plate"],AXIS_DICT["Vehicle_Manufactured_Date_Year"],AXIS_DICT["Vehicle_Reg_Date"],AXIS_DICT["Vehicle_Owner_Name"],AXIS_DICT["Vehicle_Model"],AXIS_DICT["Vehicle_Make"],AXIS_DICT["Vehicle_Type"],AXIS_DICT["Vehicle_Chassis_Number"],AXIS_DICT["Vehicle_Engine_Number"],AXIS_DICT["Vehicle_State"],1,Request_Received,Request_Proccessed,"AI",Request_Proccessed,1)

                except Exception as e:
                    print(e)
                    AXIS_DICT["Type"] = "vahaninfo"
                    # AXIS_DICT["Type"] = type
                    
                    end_numberplate_only=datetime.now()
                    total_time=(end_numberplate_only-start_universal).total_seconds()
                    
                    UID = str(claimid)
                    Claim_Warranty_Id = claimwarranty_actual
                    Dealer_Id = dealerid
                    Service_type = servicetype
                    Type = "vahaninfo"
                    Model_OCR = "In-House Model+OCR"
                    API_Request = str(API_Request)
                    API_Output = ""
                    Exception_Occurred = AXIS_DICT["Error_Message"]
                    Request_Received = str(start_universal)
                    Request_Proccessed = str(end_universal)
                    Processing_Time = total_time
                    AI_Output = ""
                    Image_name = img_name_post + ".jpg"
                    Remark = ""
                    Remark_Status = ""

                    tyreHealth_model_logs(UID, Claim_Warranty_Id, Dealer_Id, Service_type, Type, Model_OCR, API_Request, API_Output, Exception_Occurred, Request_Received, Request_Proccessed, Processing_Time, AI_Output, Image_name, Remark, Remark_Status,latitude,longitude,address,specs,ip,extra1,extra2,extra3)

                    # photo_details(Claim_Warranty_Id,"1",Image_name,Service_type,"AI",latitude,longitude,address,"AI",AI_Output,"AI", API_Output, Remark,AXIS_DICT["Error_Code"], Exception_Occurred)
                    
            except Exception as e:
                print(e)
                end_numberplate=datetime.now()
                end_universal=end_numberplate
                total_time=(end_numberplate-start_universal).total_seconds()
                total_difference=total_time
                
                UID = str(claimid)
                Claim_Warranty_Id = claimwarranty_actual
                Dealer_Id = dealerid
                Service_type = servicetype
                Type = "numberplate"
                Model_OCR = "In-House Model+OCR"
                API_Request = str(API_Request)
                API_Output = ""
                Exception_Occurred = AXIS_DICT["Error_Message"]
                Request_Received = str(start_universal)
                Request_Proccessed = str(end_universal)
                Processing_Time = total_time
                AI_Output = ""
                Image_name = img_name_post + ".jpg"
                Remark = ""
                Remark_Status = ""

                tyreHealth_model_logs(UID, Claim_Warranty_Id, Dealer_Id, Service_type, Type, Model_OCR, API_Request, API_Output, Exception_Occurred, Request_Received, Request_Proccessed, Processing_Time, AI_Output, Image_name, Remark, Remark_Status,latitude,longitude,address,specs,ip,extra1,extra2,extra3)

                photo_details(Claim_Warranty_Id,"1",Image_name,Service_type,"AI",latitude,longitude,address,"AI",AI_Output,"AI", API_Output, Remark,AXIS_DICT["Error_Code"], Exception_Occurred)
                
            AXIS_DICT["Type"] = "numberplate"
            
            return AXIS_DICT
        
        if type=="defect-outside":
            try:
                #detect tyre or not
                tyre_define=tiredetect(claimid +"/image.jpg")
                print("This is neww oldd",tyre_define)
                if tyre_define=="not_tyre":
                    return_code=244
                else:
                    return_code=""
                    print("Inside defect Model")

                start = time.time() 
                # creating pkl folders into claimid folder
                existing_directory = claimid
                new_directory_name3 = 'Splitted_Images'
                new_directory_name4 = 'Masked_Images'
                # new_directory_name5 = 'Defect_Detected_Images'
                new_directory_path3 = os.path.join(existing_directory, new_directory_name3)
                new_directory_path4 = os.path.join(existing_directory, new_directory_name4)
                # new_directory_path5 = os.path.join(existing_directory, new_directory_name5)
                
                
                os.mkdir(new_directory_path3)
                os.mkdir(new_directory_path4)
                # os.mkdir(new_directory_path5)
                print(f"Directories created successfully.")
            
                if return_code=="":
                    defect_type,return_code = find_defect_outside(claimid)
    
                # try:
                #     image_post(img_name_post , claimid , API_Request["reg_number"] ,providerId ,claimwarranty_actual)
                # except Exception as e:          
                #     print("Exception occured in posting data in first attempt")
                #     print("second attempt called")
                    
                #     try:
                #         image_post(img_name_post , claimid , API_Request["reg_number"] ,providerId ,claimwarranty_actual)
                #     except Exception as e:
                #         print("Exception occured in posting data in second attempt")
                
                if return_code==271:
                        AXIS_DICT["Error_Code"]=271
                        AXIS_DICT["Error_Message"]=error_dict["271"]
                if return_code==261:
                    AXIS_DICT["Error_Code"]=261
                    AXIS_DICT["Error_Message"]=error_dict["261"]
                if return_code==244:
                    AXIS_DICT["Error_Code"]=244
                    AXIS_DICT["Error_Message"]=error_dict["244"]

                prediction = "defect_type is:"+str(defect_type)
                prediction_=prediction
                end = time.time() - start
                print("Time taken for defect Process",end) 

                sr_one_dict={"Tyre_No": 1,
                    "Tyre_Serial":"",
                    "Photo_Name":"tyre.jpg",
                    "Remark":"",
                    "Coin_Depth_Result":"",
                    "Gauge_Result":"",
                    "Defect_Result" :str(defect_type)
                    }
                
                AXIS_DICT["Tyrewise_AI_Output"].append(sr_one_dict)
                AXIS_DICT["Type"] = type

                end_defect=datetime.now()
                end_universal=end_defect
                total_time=(end_defect-start_universal).total_seconds()
                total_difference=total_time
                
                UID = str(claimid)
                Claim_Warranty_Id = claimwarranty_actual
                Dealer_Id = dealerid
                Service_type = servicetype
                Type = type
                Model_OCR = "In-House Model"
                API_Request = str(API_Request)
                API_Output = str(AXIS_DICT)
                Exception_Occurred = AXIS_DICT["Error_Message"]
                Request_Received = str(start_universal)
                Request_Proccessed = str(end_universal)
                Processing_Time = total_time
                AI_Output = str(defect_type)
                Image_name = img_name_post + ".jpg"
                Remark = ""
                Remark_Status = ""

                tyreHealth_model_logs(UID, Claim_Warranty_Id, Dealer_Id, Service_type, Type, Model_OCR, API_Request, API_Output, Exception_Occurred, Request_Received, Request_Proccessed, Processing_Time, AI_Output, Image_name, Remark, Remark_Status,latitude,longitude,address,specs,ip,extra1,extra2,extra3)

                photo_details(Claim_Warranty_Id,"2",Image_name,Service_type,"AI",latitude,longitude,address,"AI",AI_Output,"AI", API_Output, Remark,AXIS_DICT["Error_Code"], Exception_Occurred)
                
            except Exception as e:
                print(e)
                AXIS_DICT["Type"] = type
                
                end_defect=datetime.now()
                end_universal=end_defect
                total_time=(end_defect-start_universal).total_seconds()
                total_difference=total_time
                
                UID = str(claimid)
                Claim_Warranty_Id = claimwarranty_actual
                Dealer_Id = dealerid
                Service_type = servicetype
                Type = type
                Model_OCR = "In-House Model"
                API_Request = str(API_Request)
                API_Output = ""
                Exception_Occurred = AXIS_DICT["Error_Message"]
                Request_Received = str(start_universal)
                Request_Proccessed = str(end_universal)
                Processing_Time = total_time
                AI_Output = ""
                Image_name = img_name_post + ".jpg"
                Remark = ""
                Remark_Status = ""

                tyreHealth_model_logs(UID, Claim_Warranty_Id, Dealer_Id, Service_type, Type, Model_OCR, API_Request, API_Output, Exception_Occurred, Request_Received, Request_Proccessed, Processing_Time, AI_Output, Image_name, Remark, Remark_Status,latitude,longitude,address,specs,ip,extra1,extra2,extra3)

                photo_details(Claim_Warranty_Id,"2",Image_name,Service_type,"AI",latitude,longitude,address,"AI",AI_Output,"AI", API_Output, Remark,AXIS_DICT["Error_Code"], Exception_Occurred)

            return AXIS_DICT

        if type=="meter":
            
            try:
                meter_read,return_code=meter_read_extract(str(claimid)+"/image.jpg",str(claimid))
                
                if return_code==231:
                    AXIS_DICT["Error_Code"]=231
                    AXIS_DICT["Error_Message"]=error_dict["231"]

                if return_code==200:
                    AXIS_DICT["Error_Code"]=200
                    AXIS_DICT["Error_Message"]=error_dict["200"]

               
                AXIS_DICT["meter_read"]=str(meter_read)

                sr_one_dict={"Tyre_No": 1,
                    "Tyre_Serial":"",
                    "Photo_Name":"tyre.jpg",
                    "Remark":"",
                    "Coin_Depth_Result":"",
                    "Gauge_Result":"",
                    "Defect_Result" :""
                    }

                AXIS_DICT["Tyrewise_AI_Output"].append(sr_one_dict)

                end_numberplate=datetime.now()
                end_universal=end_numberplate
                total_time=(end_numberplate-start_universal).total_seconds()
                total_difference=total_time
                
                UID = str(claimid)
                Claim_Warranty_Id = claimwarranty_actual
                Dealer_Id = dealerid
                Service_type = servicetype
                Type = "meter"
                Model_OCR = "In-House Model+OCR"
                API_Request = str(API_Request)
                API_Output = str(AXIS_DICT)
                Exception_Occurred = AXIS_DICT["Error_Message"]
                Request_Received = str(start_universal)
                Request_Proccessed = str(end_universal)
                Processing_Time = total_time
                AI_Output = str(meter_read)
                Image_name = img_name_post + ".jpg"
                Remark = ""
                Remark_Status = ""

                tyreHealth_model_logs(UID, Claim_Warranty_Id, Dealer_Id, Service_type, Type, Model_OCR, API_Request, API_Output, Exception_Occurred, Request_Received, Request_Proccessed, Processing_Time, AI_Output, Image_name, Remark, Remark_Status,latitude,longitude,address,specs,ip,extra1,extra2,extra3)
                
            except Exception as e:
                print(e)
                end_numberplate=datetime.now()
                end_universal=end_numberplate
                total_time=(end_numberplate-start_universal).total_seconds()
                total_difference=total_time
                
                UID = str(claimid)
                Claim_Warranty_Id = claimwarranty_actual
                Dealer_Id = dealerid
                Service_type = servicetype
                Type = "meter"
                Model_OCR = "In-House Model+OCR"
                API_Request = str(API_Request)
                API_Output = ""
                Exception_Occurred = AXIS_DICT["Error_Message"]
                Request_Received = str(start_universal)
                Request_Proccessed = str(end_universal)
                Processing_Time = total_time
                AI_Output = ""
                Image_name = img_name_post + ".jpg"
                Remark = ""
                Remark_Status = ""

                tyreHealth_model_logs(UID, Claim_Warranty_Id, Dealer_Id, Service_type, Type, Model_OCR, API_Request, API_Output, Exception_Occurred, Request_Received, Request_Proccessed, Processing_Time, AI_Output, Image_name, Remark, Remark_Status,latitude,longitude,address,specs,ip,extra1,extra2,extra3)
            
            return AXIS_DICT
        
        if type=="gas_meter":
            
            try:
                meter_read,return_code=gas_meter_reading_extract(str(claimid)+"/image.jpg",str(claimid))
                
                if return_code==231:
                    AXIS_DICT["Error_Code"]=231
                    AXIS_DICT["Error_Message"]=error_dict["231"]

                if return_code==200:
                    AXIS_DICT["Error_Code"]=200
                    AXIS_DICT["Error_Message"]=error_dict["200"]

               
                AXIS_DICT["gas_meter_read"]=str(meter_read)

                sr_one_dict={"Tyre_No": 1,
                    "Tyre_Serial":"",
                    "Photo_Name":"tyre.jpg",
                    "Remark":"",
                    "Coin_Depth_Result":"",
                    "Gauge_Result":"",
                    "Defect_Result" :""
                    }
                AXIS_DICT["Tyrewise_AI_Output"].append(sr_one_dict)

                end_numberplate=datetime.now()
                end_universal=end_numberplate
                total_time=(end_numberplate-start_universal).total_seconds()
                total_difference=total_time
                
                UID = str(claimid)
                Claim_Warranty_Id = claimwarranty_actual
                Dealer_Id = dealerid
                Service_type = servicetype
                Type = "gas_meter"
                Model_OCR = "In-House Model+OCR"
                API_Request = str(API_Request)
                API_Output = str(AXIS_DICT)
                Exception_Occurred = AXIS_DICT["Error_Message"]
                Request_Received = str(start_universal)
                Request_Proccessed = str(end_universal)
                Processing_Time = total_time
                AI_Output = str(meter_read)
                Image_name = img_name_post + ".jpg"
                Remark = ""
                Remark_Status = ""

                tyreHealth_model_logs(UID, Claim_Warranty_Id, Dealer_Id, Service_type, Type, Model_OCR, API_Request, API_Output, Exception_Occurred, Request_Received, Request_Proccessed, Processing_Time, AI_Output, Image_name, Remark, Remark_Status,latitude,longitude,address,specs,ip,extra1,extra2,extra3)
                
            except Exception as e:
                print(e)
                end_numberplate=datetime.now()
                end_universal=end_numberplate
                total_time=(end_numberplate-start_universal).total_seconds()
                total_difference=total_time
                
                UID = str(claimid)
                Claim_Warranty_Id = claimwarranty_actual
                Dealer_Id = dealerid
                Service_type = servicetype
                Type = "meter"
                Model_OCR = "In-House Model+OCR"
                API_Request = str(API_Request)
                API_Output = ""
                Exception_Occurred = AXIS_DICT["Error_Message"]
                Request_Received = str(start_universal)
                Request_Proccessed = str(end_universal)
                Processing_Time = total_time
                AI_Output = ""
                Image_name = img_name_post + ".jpg"
                Remark = ""
                Remark_Status = ""

                tyreHealth_model_logs(UID, Claim_Warranty_Id, Dealer_Id, Service_type, Type, Model_OCR, API_Request, API_Output, Exception_Occurred, Request_Received, Request_Proccessed, Processing_Time, AI_Output, Image_name, Remark, Remark_Status,latitude,longitude,address,specs,ip,extra1,extra2,extra3)
            
            return AXIS_DICT

    except Exception as e:
        try:
            AXIS_DICT = dict()
            AXIS_DICT=claim_main[claimid]
            AXIS_DICT["Tyre_Unqiue_Id"]=claimwarranty_actual
            AXIS_DICT["Type"]=type
            print("exception occured on whole flow due to >>>",e)
            tyreHealth_model_logs("", claimid, dealerid, servicetype, type, "", "", "", str(e), datetime.today(), "", 0.0, "", "", "", "",latitude,longitude,address,specs,ip,extra1,extra2,extra3)
            AXIS_DICT["Error_Code"] = 422
            AXIS_DICT["Error_Message"] = "Unprocessable Entity"
            return AXIS_DICT
        
        except Exception as e:
            AXIS_DICT = dict()
            print(e)
            AXIS_DICT["Error_Code"] = 422
            AXIS_DICT["Error_Message"] = "Unprocessable Entity"
            return AXIS_DICT
    

if __name__ == "__main__":
    uvicorn.run(app, debug=True,port=443)
    # uvicorn.run(app, host="0.0.0.0",port=8089)