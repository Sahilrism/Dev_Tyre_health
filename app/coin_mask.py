from PIL import Image, ImageDraw
import cv2,codecs,json

# lead = "/home/deep/Desktop/Tyre_Health_Dev/Coin_leads/22e5ceb4-9fe6-44c7-87d0-cdc43135c9a3_coin"
# leadsid = "/root/Downloads/final_repo/Tyre_Health/Claim_ID/Coin_Leads/"+lead

def draw_img(claim_id):
    image = Image.open(claim_id + "/image.jpg")
    draw = ImageDraw.Draw(image)
    # load predicted mask into an object
    file_path = claim_id + "/mask_coord.json"
    obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
    b_new = json.loads(obj_text)
    points = b_new
    for point in points:
        draw.point(point, fill="red")
    # Save the image
    image.save(claim_id + "/mask.jpg")
    
# draw_img(lead)
