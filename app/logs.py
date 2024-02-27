import pymysql

def tyreHealth_model_logs(UID, Claim_Warranty_Id, Dealer_Id, Service_type, Type, Model_OCR, API_Request, API_Output, Exception_Occurred, Request_Received, Request_Proccessed, Processing_Time, AI_Output, Image_name, Remark, Remark_Status,latitude,longitude,address,specs,ip,extra1,extra2,extra3):
    # To connect the sql server
    pass
    con = pymysql.connect(host="164.52.209.14", user="root", passwd="Check@2020", db="tyre_health_uat",port=3306)
    # con = pymysql.connect(host="103.248.83.79", user="mysqladmin", passwd="sql@#cloud@ce", db="tyre_healthcheck_uat",port=3306)
    cursor = con.cursor()
    print("Connection build")
    # SP Calling
    cursor.callproc('USP_FillDetails',(UID, Claim_Warranty_Id, Dealer_Id, Service_type, Type, Model_OCR, API_Request, API_Output, Exception_Occurred, Request_Received, Request_Proccessed, Processing_Time, AI_Output, Image_name, Remark, Remark_Status,latitude,longitude,address,specs,ip,extra1,extra2,extra3))

    print("Data inserted in table api_tyrecheck_model_logs")
    con.commit()
    con.close()
    

def photo_details(Tyre_HealthCheckId,category_Id,PhotoName,ServiceType,Photo_URL,latitude,Longitude,Photo_Capture_Address,Photo_UploadBySyatem,AIResult,CreatedBy, AIResult_Json, Remarks,Error_Code, Error_Message):
    # To connect the sql server
    pass
    con = pymysql.connect(host="164.52.209.14", user="root", passwd="Check@2020", db="tyre_health_uat",port=3306)
    # con = pymysql.connect(host="103.248.83.79", user="mysqladmin", passwd="sql@#cloud@ce", db="tyre_healthcheck_uat",port=3306)
    cursor = con.cursor()
    print("Connection build")
    # SP Calling
    cursor.callproc('usp_Insert_PhotosDetails',(Tyre_HealthCheckId,category_Id,PhotoName,ServiceType,Photo_URL,latitude,Longitude,Photo_Capture_Address,Photo_UploadBySyatem,AIResult,CreatedBy, AIResult_Json, Remarks,Error_Code, Error_Message))

    print("Data inserted in table Photo Details")
    con.commit()
    con.close()

def vehicle_details(Tyre_HealthCheckId,Type,Vehicle_Number_Plate,Vehicle_Manufactured_Date_Year,Vehicle_Reg_Date,Vehicle_Owner_Name,Vehicle_Model,Vehicle_Make,Vehicle_Type,Vehicle_Chassis_Number,Vehicle_Engine_Number,Vehicle_State,pdf_generated,StateTime,EndTime):
    return 0

    # To connect the sql server
    con = pymysql.connect(host="164.52.209.14", user="root", passwd="Check@2020", db="tyre_health_uat",port=3306)
    # con = pymysql.connect(host="103.248.83.79", user="mysqladmin", passwd="sql@#cloud@ce", db="tyre_healthcheck_uat",port=3306)
    cursor = con.cursor()
    print("Connection build")
    # SP Calling
    cursor.callproc('usp_Insert_Customer_Vehicle_Tyre_Details',(Tyre_HealthCheckId,Type,Vehicle_Number_Plate,Vehicle_Manufactured_Date_Year,Vehicle_Reg_Date,Vehicle_Owner_Name,Vehicle_Model,Vehicle_Make,Vehicle_Type,Vehicle_Chassis_Number,Vehicle_Engine_Number,Vehicle_State,pdf_generated,StateTime,EndTime))

    print("Data inserted in table Vehicle Details")
    con.commit()
    con.close()


def dealers():
    pass
    con = pymysql.connect(host="164.52.209.14", user="root", passwd="Check@2020", db="tyre_health_uat",port=3306)
    # con = pymysql.connect(host="103.248.83.79", user="mysqladmin", passwd="sql@#cloud@ce", db="tyre_healthcheck_uat",port=3306)
    cursor = con.cursor()
    # print("Connection build")

    # Creating a cursor object using the cursor() method
    cursor = con.cursor()

    #Executing the query
    cursor.callproc('GetDealers',())
    record = cursor.fetchall()
    con.commit()
    con.close()
    dealer_list = []
    for i in record:
        for j in i:
            dealer_list.append(j)
    # print("Dealer_Codes : ",dealer_list)
    return dealer_list

# def vehicle_details(Claim_warranty_id,Type, Service_type ,Vehicle_Number_Plate ,Vehicle_Owner_Name ,Vehicle_Manufactured_On,Vehicle_Reg_On,Vehicle_Chassis_Number ,Vehicle_Engine_Number ,Vehicle_Make ,Vehicle_Model ,Vehicle_Type,Requested_DateTime):
#     # To connect the sql server
#     con = pymysql.connect(host="103.248.83.79", user="mysqladmin", passwd="mysqladmin@123!", db="tyre_healthcheck_uat",port=3306)
#     cursor = con.cursor()
#     print("Connection build")
#     # SP Calling
#     cursor.callproc('USP_Insert_VehDetails_Api',(Claim_warranty_id,Type, Service_type ,Vehicle_Number_Plate ,Vehicle_Owner_Name ,Vehicle_Manufactured_On,Vehicle_Reg_On,Vehicle_Chassis_Number ,Vehicle_Engine_Number ,Vehicle_Make ,Vehicle_Model ,Vehicle_Type,Requested_DateTime))

#     print("Data inserted in table Vehicle Details")
#     con.commit()
#     con.close()

def get_out_defect(claim_id,type_):
    return "Good"
    con = pymysql.connect(host="164.52.209.14", user="root", passwd="Check@2020", db="tyre_health_uat",port=3306)
    # con = pymysql.connect(host="103.248.83.79", user="mysqladmin", passwd="sql@#cloud@ce", db="tyre_healthcheck_uat",port=3306)
    cursor = con.cursor()
    print("Connection build")
    # Creating a cursor object using the cursor() method
    cursor = con.cursor()
    #Executing the query
    cursor.callproc('Verify_OutsideDefect',(claim_id,type_))
    record = cursor.fetchall()
    con.commit()
    con.close()
    print("record in database for outside image : ",record)
    # if len(record)<=0:
    #     return ""
    # else:
    #     return record[0][0]
    
    return record[0][0]

def settingcheck():
    pass
    # con = pymysql.connect(host="103.248.83.79", user="mysqladmin", passwd="sql@#cloud@ce", db="tyre_healthcheck_uat",port=3306)
    con = pymysql.connect(host="164.52.209.14", user="root", passwd="Check@2020", db="tyre_health_uat",port=3306)
    cursor = con.cursor()
    print("Connection build")

    # Creating a cursor object using the cursor() method
    cursor = con.cursor()

    #Executing the query
    cursor.callproc('use_TyreHealthNewId',())
    record = cursor.fetchall()
    # print("Thisiisiisxxxxxxx",record)
    con.commit()
    con.close()
    print("Tyre Health APP Checked")
    # print(record)
    return record

def settingchecknew():
    pass
    # con = pymysql.connect(host="103.248.83.79", user="mysqladmin", passwd="sql@#cloud@ce", db="tyre_healthcheck_uat",port=3306)
    con = pymysql.connect(host="164.52.209.14", user="root", passwd="Check@2020", db="tyre_health_uat",port=3306)
    cursor = con.cursor()
    print("Connection build")

    # Creating a cursor object using the cursor() method
    cursor = con.cursor()

    #Executing the query
    cursor.callproc('SettingssFetch',())
    record = cursor.fetchall()
    con.commit()
    con.close()
    print("Tyre Health Stencil Setting Checked")
    # print(record)
    return record

def tyre_logs(Claim_ID, apiRequestDate, apiResponse):
    pass
    # To connect the sql server
    con = pymysql.connect(host="164.52.209.14", user="root", passwd="Check@2020", db="tyre_health_uat",port=3306)
    # con = pymysql.connect(host="103.248.83.79", user="mysqladmin", passwd="sql@#cloud@ce", db="tyre_healthcheck_uat",port=3306)
    cursor = con.cursor()
    print("Connection build for tyre log")
    # SP Calling
    cursor.callproc('USP_logDetails',(Claim_ID, apiRequestDate,apiResponse))

    print("Data inserted in table tyre logs")
    con.commit()
    con.close()

def coin_size():
    pass
    #Connection building
    con = pymysql.connect(host="164.52.209.14", user="root", passwd="Check@2020", db="tyre_health_uat",port=3306)
    cursor = con.cursor()
    print("Connection build")

    # Creating a cursor object using the cursor() method
    cursor = con.cursor()

    #Executing the query
    cursor.callproc('fetch_coin_size',())

    #Fetching the data from db
    record = cursor.fetchall()
    con.commit()
    con.close()
    return record