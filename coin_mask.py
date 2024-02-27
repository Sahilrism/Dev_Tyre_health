from PIL import Image, ImageDraw
import cv2,codecs,json

lead = "/home/deep/Desktop/Tyre_Health_Dev/Claim_ID/7e33947e-f97d-49e9-a851-7a568b9a7be1_coin"
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
    image.save(claim_id + "/image_detected.jpg")
    
draw_img(lead)
