import requests,glob


def image_post(type_ , claim_id , reg_no ,provider_id ,warranty_id):
   

    # url = "https://bridgestone.checkexplore.com/VCWebAPI/api/AI/PostFileDetailFromAI?Reg_Number="+ reg_no + "&agencyId="+provider_id+"&strPhotoName="+type_+".jpg+"+"&claimid=" + warranty_id

    url = "https://tyrehealth.checkexplore.com/VCWebAPI/api/AI/PostFileDetailFromAI?Reg_Number="+ reg_no + "&agencyId="+provider_id+"&strPhotoName="+type_+".jpg+"+"&claimid=" + warranty_id

    payload={}
    file_loc_ = glob.glob(claim_id+"/*.jpg")
    # print("Thisssssssssssssssssssfile",file_loc_)
    file_loc = file_loc_[0]
    for i in file_loc_:
        if ("image.jpg" not in i) or ("image.jpeg" not in i) or ("image.png" not in i):
            file_loc = i
            print("file location for posting images === ",i) 
    if "serial" in type_:
        files=[
            ('img',(type_,open(file_loc,'rb'),'image/jpeg'))
            ]
    else:
        files=[
        ('img',(type_+'.jpg',open(claim_id+'/image.jpg','rb'),'image/jpeg'))
        ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    print(response.text)
    # # Set the API endpoint and parameters
    # endpoint = 'https://dev.checkexplore.com/VCWebAPI/api/AI/PostFileDetailFromAI/'
    # params = {'Reg_Number': reg_no, 'agencyId': provider_id,'strPhotoName':'claimid':''}

    # # Make a GET request to the API
    # response = requests.get(endpoint, params=params)

    # # Print the API response
    # print(response.text)

def new_image(type_ , claim_id , reg_no ,provider_id ,warranty_id):

    url = "https://tyrehealth.checkexplore.com/VCWebAPI/api/AI/PostFileDetailFromAI?Reg_Number="+ reg_no + "&agencyId="+provider_id+"&strPhotoName="+type_+".jpg+"+"&claimid=" + warranty_id

    payload={}
    file_loc_ = glob.glob(claim_id+"/*.jpg")
    print("Thisssssssssssssssssssfilenew",file_loc_)
    file_loc = file_loc_[0]
    for i in file_loc_:
        if ("image.jpg" not in i) or ("image.jpeg" not in i) or ("image.png" not in i):
            file_loc = i
            print("file location for posting images === ",i) 
    # if "serial" in type_:
    #     files=[
    #         ('img',(type_,open(file_loc,'rb'),'image/jpeg'))
    #         ]
    else:
        files=[
        ('img',(type_+'.jpg',open(claim_id+'/mask.jpg','rb'),'mask/jpeg'))
            
        ]
        print("mASK image posted")
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    print(response.text)