import requests,re
from app.regex_num import number_regerx
from PIL import Image
import pymysql,uuid
from datetime import datetime
from app.logs import *



def api_info(number,claimid,API_Request):
  number = number.replace(" ","")
  return_code=""
  uid_unique=str(uuid.uuid4())
  vahan_final={}
  vahan_final[uid_unique]={}

  veh_dict=vahan_final[uid_unique]
  veh_dict["Vehicle_Owner_Name"] = ""
  veh_dict["Vehicle_father_name"] = ""
  veh_dict["Vehicle_Number_Plate"] = ""
  veh_dict["Vehicle_Reg_Date"] = ""
  veh_dict["Vehicle_Manufactured_Date_Year"] = ""
  veh_dict["Vehicle_presentAddress"] = ""
  veh_dict["Vehicle_addressLine"] = ""
  veh_dict["Vehicle_country"] = ""
  veh_dict["Vehicle_state_cd"] = ""
  veh_dict["Vehicle_State"] = ""
  veh_dict["Vehicle_district_name"] = ""
  veh_dict["Vehicle_city_name"] = ""
  veh_dict["Vehicle_pincode"] = ""
  veh_dict["Vehicle_Type"] = ""
  veh_dict["Vehicle_Make"] = ""
  veh_dict["Vehicle_Model"] = ""
  veh_dict["Vehicle_variant"] = ""
  veh_dict["Vehicle_Chassis_Number"] = ""
  veh_dict["Vehicle_Engine_Number"] = ""
  veh_dict["Vehicle_vehicle_cd"] = ""
  veh_dict["Vehicle_status"] = ""
  veh_dict["Vehicle_timestamp"] = ""
  veh_dict["Vehicle_vehicleManufacturerName"] = ""
  veh_dict["Vehicle_fuel_type"] = ""
  veh_dict["Vehicle_normsType"] = ""
  veh_dict["Vehicle_bodyType"] = ""
  veh_dict["Vehicle_ownerCount"] = ""
  veh_dict["Vehicle_statusAsOn"] = ""
  veh_dict["Vehicle_regAuthority"] = ""
  veh_dict["Vehicle_rcExpiryDate"] = ""
  veh_dict["Vehicle_vehicleTaxUpto"] = ""
  veh_dict["Vehicle_vehicleInsuranceCompanyName"] = ""
  veh_dict["Vehicle_vehicleInsuranceUpto"] = ""
  veh_dict["Vehicle_vehicleInsurancePolicyNumber"] = ""
  veh_dict["Vehicle_rcFinancer"] = ""
  veh_dict["Vehicle_vehicleCubicCapacity"] = ""
  veh_dict["Vehicle_unladenWeight"] = ""
  veh_dict["Vehicle_vehicleCylindersNo"] = ""
  veh_dict["Vehicle_vehicleSeatCapacity"] = ""
  veh_dict["Vehicle_puccNumber"] = ""
  veh_dict["Vehicle_puccUpto"] = ""
  veh_dict["Vehicle_isCommercial"] = ""


  service_start=0
  print("Reg No. to vahan API : ",number)
  if type(number) is not str:
    number = number[0]

  try:
      start=datetime.now()
      service_start=start
      
      url = "http://dev.checkexplore.com:93/VCWebAPI/api/AI/vehicle_details?registration_no="+number
      
      info_txt=requests.request("POST", url)
      info_txt=info_txt.json()
      print(info_txt)
      
      
      # veh_dict["Vehicle_Number_Plate"]=info_txt["registration_no"]
      # veh_dict["Vehicle_Owner_Name"]=info_txt["owner_name"]
      # veh_dict["Vehicle_Manufactured_Date_Year"]=info_txt["manufacturer_date"]
      # veh_dict["Vehicle_Reg_Date"]=info_txt["registration_date"]
      # veh_dict["Vehicle_Make"]=info_txt["make"]
      # veh_dict["Vehicle_Model"]=info_txt["model"]
      # veh_dict["Vehicle_Type"]=info_txt["vehicletype"]
      # veh_dict["Vehicle_Chassis_Number"]=info_txt["chassis_no"]
      # veh_dict["Vehicle_Engine_Number"]=info_txt["engine_no"]
      # veh_dict["Vehicle_State"]=info_txt["state_cd"]
      
      veh_dict["Vehicle_Owner_Name"] = info_txt["owner_name"]
      veh_dict["Vehicle_father_name"] = info_txt["father_name"]
      veh_dict["Vehicle_Number_Plate"] = info_txt["registration_no"]
      veh_dict["Vehicle_Reg_Date"] = info_txt["registration_date"]
      veh_dict["Vehicle_Manufactured_Date_Year"] = info_txt["manufacturer_date"]
      veh_dict["Vehicle_presentAddress"] = info_txt["presentAddress"]
      veh_dict["Vehicle_addressLine"] = info_txt["addressLine"]
      veh_dict["Vehicle_country"] = info_txt["country"]
      veh_dict["Vehicle_state_cd"] = info_txt["state_cd"]
      veh_dict["Vehicle_State"] = info_txt["state_name"]
      veh_dict["Vehicle_district_name"] = info_txt["district_name"]
      veh_dict["Vehicle_city_name"] = info_txt["city_name"]
      veh_dict["Vehicle_pincode"] = info_txt["pincode"]
      veh_dict["Vehicle_Type"] = info_txt["vehicletype"]
      veh_dict["Vehicle_Make"] = info_txt["make"]
      veh_dict["Vehicle_Model"] = info_txt["model"]
      veh_dict["Vehicle_variant"] = info_txt["variant"]
      veh_dict["Vehicle_Chassis_Number"] = info_txt["chassis_no"]
      veh_dict["Vehicle_Engine_Number"] = info_txt["engine_no"]
      veh_dict["Vehicle_cd"] = info_txt["vehicle_cd"]
      veh_dict["Vehicle_status"] = info_txt["status"]
      veh_dict["Vehicle_timestamp"] = info_txt["timestamp"]
      veh_dict["Vehicle_ManufacturerName"] = info_txt["vehicleManufacturerName"]
      veh_dict["Vehicle_fuel_type"] = info_txt["type"]
      veh_dict["Vehicle_normsType"] = info_txt["normsType"]
      veh_dict["Vehicle_bodyType"] = info_txt["bodyType"]
      veh_dict["Vehicle_ownerCount"] = info_txt["ownerCount"]
      veh_dict["Vehicle_statusAsOn"] = info_txt["statusAsOn"]
      veh_dict["Vehicle_regAuthority"] = info_txt["regAuthority"]
      veh_dict["Vehicle_rcExpiryDate"] = info_txt["rcExpiryDate"]
      veh_dict["Vehicle_TaxUpto"] = info_txt["vehicleTaxUpto"]
      veh_dict["Vehicle_InsuranceCompanyName"] = info_txt["vehicleInsuranceCompanyName"]
      veh_dict["Vehicle_InsuranceUpto"] = info_txt["vehicleInsuranceUpto"]
      veh_dict["Vehicle_InsurancePolicyNumber"] = info_txt["vehicleInsurancePolicyNumber"]
      veh_dict["Vehicle_rcFinancer"] = info_txt["rcFinancer"]
      veh_dict["Vehicle_CubicCapacity"] = info_txt["vehicleCubicCapacity"]
      veh_dict["Vehicle_unladenWeight"] = info_txt["unladenWeight"]
      veh_dict["Vehicle_CylindersNo"] = info_txt["vehicleCylindersNo"]
      veh_dict["Vehicle_SeatCapacity"] = info_txt["vehicleSeatCapacity"]
      veh_dict["Vehicle_puccNumber"] = info_txt["puccNumber"]
      veh_dict["Vehicle_puccUpto"] = info_txt["puccUpto"]
      veh_dict["Vehicle_isCommercial"] = info_txt["isCommercial"]


      end=datetime.now()
      total_time=(end-start).total_seconds()
      print("*"*100)
      print(info_txt)
      print("*"*100)
      if veh_dict["Vehicle_Owner_Name"]=="" or veh_dict["Vehicle_Owner_Name"]==None:
        return_code = 203

  except Exception as e:
      print(e)
      print("Numberplate API Exception")
      return_code=203
      end_exception=datetime.now()
      process_time=(end_exception-service_start).total_seconds()

  return veh_dict,return_code
