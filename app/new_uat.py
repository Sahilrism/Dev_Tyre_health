import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random,threading,shutil,uuid,glob
from PIL import Image
import matplotlib.pyplot as plt
import colorsys
from skimage.measure import find_contours as fc
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon as p
from matplotlib.patches import Polygon

from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow
from tensorflow.keras.models import load_model
from imutils import build_montages
from imutils import paths
import argparse
import pickle,time

from app.eval_tyre_mask import tyre_mask_func
from app.tyre_mask_defect_intersect import tyre_mask_defect

def configuration_tyre_detection():
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"))
    
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8# set a custom testing    
    
    cfg.MODEL.WEIGHTS = 'app/tyre_crop.pth'
    predictor = DefaultPredictor(cfg)
    return predictor

predict_tyre = configuration_tyre_detection()

def tyre_detection(claimid):

    img_path = claimid + "/image.jpg"
    clss=["obj","tyre"]
    name=img_path.split("/")[-1]
    im=cv2.imread(img_path)
    output=predict_tyre(im)
    print("This is Cropped tyre Output: ", output)
    pred_box = output["instances"].pred_boxes.tensor.cpu().numpy()
    pred_class = output["instances"].pred_classes.cpu().numpy()
    pred_score = output["instances"].scores.cpu().numpy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # org
    org = (50, 50)
    
    # fontScale
    fontScale = 0.5
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 1

    pred_box_=list(pred_box)
    pred_score=list(pred_score)

    if len(pred_box) > 0 :
        max_thr_op=pred_score.index(max(pred_score))
        pred_box_=pred_box[max_thr_op]
        pred_score = output["instances"].scores.cpu().numpy()
        pred_box_=list(pred_box_)

        for i in range(len(pred_box_)):
            cropped_tyre_image = im[int(pred_box_[1]):int(pred_box_[3]), int(pred_box_[0]):int(pred_box_[2])]
            cv2.imwrite(claimid+"/image.jpg",cropped_tyre_image)

    else:
        print("Tyre not identified by model")

def crop_image(claimid):
    
    # Prompt the user to select an image file
    file_path = claimid + "/image.jpg"
    image_ = cv2.imread(file_path)
    print(cv2.imread(file_path).shape)
    height,width,ax = cv2.imread(file_path).shape
    new_width = width+0
    new_height = height+0
    # Add a single unit to width and height
    if height%2!=0:
        new_height = height + 1
    if width%2!=0:
        new_width = width + 1
    resized_image = cv2.resize(image_, (new_width, new_height))
    cv2.imwrite(claimid + '/image.jpg', resized_image)
    
    # Check if a file was selected
    if file_path:
        # Read the image using OpenCV
        image = cv2.imread(file_path)

        # Check if the image was successfully loaded
        if image is not None:
            # Create a directory to store the cropped images
            output_dir = claimid + "/Splitted_Images"

            # Get image dimensions
            height, width, _ = image.shape

            # Calculate the coordinates for cropping
            half_width = width // 2
            half_height = height // 2

            # Crop the image into four parts
            top_left = image[0:half_height, 0:half_width]
            top_right = image[0:half_height, half_width:width]
            bottom_left = image[half_height:height, 0:half_width]
            bottom_right = image[half_height:height, half_width:width]

            # Save the cropped images
            cv2.imwrite(os.path.join(output_dir, "top_left.jpg"), top_left)
            cv2.imwrite(os.path.join(output_dir, "top_right.jpg"), top_right)
            cv2.imwrite(os.path.join(output_dir, "bottom_left.jpg"), bottom_left)
            cv2.imwrite(os.path.join(output_dir, "bottom_right.jpg"), bottom_right)

            print("Cropped images saved successfully.")
        else:
            print("Failed to load the image.")
    else:
        print("No file selected.")

def merge_image(claimid):
    # Load the four cropped images
    top_left = cv2.imread(claimid + "/Masked_Images" + "/top_left.jpg")
    top_right = cv2.imread(claimid + "/Masked_Images" + "/top_right.jpg")
    bottom_left = cv2.imread(claimid + "/Masked_Images" + "/bottom_left.jpg")
    bottom_right = cv2.imread(claimid + "/Masked_Images" + "/bottom_right.jpg")

    # Check if the images were successfully loaded
    if all(img is not None for img in [top_left, top_right, bottom_left, bottom_right]):
        # Get the dimensions of the cropped images
        height, width, _ = top_left.shape

        # Adjust the dimensions of the bottom-left and bottom-right images
        bottom_left = bottom_left[:height, :]
        bottom_right = bottom_right[:height, :]

        # Create a blank canvas to merge the images
        merged_image = np.zeros((2 * height, 2 * width, 3), dtype=np.uint8)

        # Place the cropped images onto the merged image
        merged_image[0:height, 0:width] = top_left
        merged_image[0:height, width:(2 * width)] = top_right
        merged_image[height:(2 * height), 0:width] = bottom_left
        merged_image[height:(2 * height), width:(2 * width)] = bottom_right

        # Save the merged image
        cv2.imwrite(claimid + "/merged_image.jpg", merged_image)
        # os.remove(claimid + "/top_left.jpg")
        # os.remove(claimid + "/top_right.jpg")
        # os.remove(claimid + "/bottom_left.jpg")
        # os.remove(claimid + "/bottom_right.jpg")

        print("Merged image saved successfully.")
    else:
        print("Failed to load one or more of the cropped images.")

def high_priority_value(input_list, priority_list):
    for value in priority_list:
        if value in input_list:
            return value
    return ""

def random_colors(N, bright=True):
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

def draw_mask_image(claimid,img,pred_class,pred_score,pred_box,pred_mask,img_path,label_):
    name = img_path.split("/")[-1].split(".")[0]
    imm = Image.open(img_path)
    fig = plt.figure(frameon=False)
    w, h = imm.size
    print(h, w)
    fig.set_size_inches(w / 100, h / 100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(imm, aspect='auto')
    length_of_pred_class = len(pred_class)
    print(length_of_pred_class)
    colors = random_colors(length_of_pred_class)
    pol_list=[]
    cord_array = []
    if length_of_pred_class != 0:
        for i in range(length_of_pred_class):
            color = colors[i]
            score = pred_score[i]
            x1, y1, x2, y2 = pred_box[i]
            label = label_
            # caption = "{} {:.3f}".format(label, score) if score else label
            caption = "{}".format(score*100)
            ax.text(x1, y1, caption, color='w', size=11, backgroundcolor="none")
            mask = pred_mask[i]
            p_m = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            p_m[1:-1, 1:-1] = mask
            con = fc(p_m, 0.5)
            for verts in con:
                polCen = None
                verts = np.fliplr(verts) - 1
                pol_cor = p(verts)
                poly_plot = Polygon(verts, edgecolor="red", facecolor='none',linewidth=5.0)
                polCen = [pol_cor.centroid.x, pol_cor.centroid.y]
                pol_list.append(polCen)
                ax.add_patch(poly_plot)
            
            modified_list = [[1 if item else 0 for item in sub_list] for sub_list in mask]
            masks_list=np.array(modified_list).flatten().tolist()
            mask_array = np.array(masks_list)

            height, width = img.shape[:2]
            # create arrays of x-coordinates and y-coordinates
            x_coords = np.arange(width)
            y_coords = np.arange(height)
            # use numpy.meshgrid to create coordinate arrays
            X, Y = np.meshgrid(x_coords, y_coords)
            # stack the X and Y arrays and convert to a list of coordinates
            im_array = np.stack((X, Y), axis=-1).reshape(-1, 2).tolist()
            im_array = np.array(im_array)
            cord_array = im_array[mask_array == 1].tolist()

    spliting = img_path.split("/")[-1]
    name, extension = os.path.splitext(spliting)
    new_name = name + extension
    plt.savefig(claimid + "/Masked_Images/" + new_name)
    return cord_array

def configuration_model_side():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8 # set a custom testing    
    cfg.MODEL.WEIGHTS="app/defect_model_side.pth"
    predictor = DefaultPredictor(cfg)
    return predictor
wear_side_predict = configuration_model_side()

def damage_define(claimid,img,path,model_config,class_names):
    img_path = path
    name = img_path.split("/")[-1].split(".")[0]
    outputs1=model_config(img)
    pred_class = outputs1["instances"].pred_classes.cpu().numpy()
    pred_score = outputs1["instances"].scores.cpu().numpy()
    pred_box = outputs1["instances"].pred_boxes.tensor.cpu().numpy()
    pred_mask_damage = outputs1["instances"].pred_masks.cpu().numpy()
    labels = []
    label_wear = ""
    defect_mask=[]
    defect_dict_ = dict()
    if len(pred_class)>0:
        for i in range(len(pred_class)):          
            label_wear = class_names[pred_class[i]]
            labels.append(label_wear)
            if ("Runflat" not in label_wear):
                defect_mask = draw_mask_image(claimid,img,pred_class,pred_score,pred_box,pred_mask_damage,img_path,label_wear)
                defect_dict_[label_wear] = defect_mask
            else:
                cv2.imwrite(claimid+"/Masked_Images/{}.jpg".format(name),img)
    else:
        cv2.imwrite(claimid+"/Masked_Images/{}.jpg".format(name),img)

    print(labels)
    return labels,defect_dict_


def find_defect_outside(claimid):
    
    defect_dict = dict()
    finaldefect = []
    defect_cls2 = ""
    
    # detect tire from whole image and crop it
    tyre_detection(claimid)
    
    # split tire into 4 parts
    crop_image(claimid)
 
    
    def defect_thread():
        # images = glob.glob(claimid + "/Splitted_Images/*.jpg")
        top_left = claimid + "/Splitted_Images/top_left.jpg"
        top_right = claimid + "/Splitted_Images/top_right.jpg"
        bottom_left = claimid + "/Splitted_Images/bottom_left.jpg"
        bottom_right = claimid + "/Splitted_Images/bottom_right.jpg"
        
        # To detect defect on the tyre using splitted images
        def defect_detect(path):
            img_path = path
            if "image.jpg" not in img_path:
                name = img_path.split("/")[-1].split(".")[0]
                img=cv2.imread(img_path)
                class_names_side=["CBU","Bead Damage","Shoulder Cut","Runflat","Sidewall Cut"]
                defect_cls = ""
                return_defect = ""  
                damage_list,defect_mask = damage_define(claimid,img,img_path,wear_side_predict,class_names_side)
                if len(damage_list) > 0:
                    order = ["Sidewall Cut","Shoulder Cut"]
                    defect_cls = high_priority_value(damage_list, order)
                    if (len(defect_mask)>0) and (defect_cls in order):
                        defect_dict[name] = defect_mask
                    finaldefect.append(defect_cls)
        start = time.time()
        t1 = threading.Thread(target=defect_detect,args=(top_left,))
        t2 = threading.Thread(target=defect_detect,args=(top_right,))
        t3 = threading.Thread(target=defect_detect,args=(bottom_left,))
        t4 = threading.Thread(target=defect_detect,args=(bottom_right,))
        
        t1.start()
        t2.start()
        t3.start()
        t4.start()

        t1.join()
        t2.join()
        t3.join()
        t4.join()
        
        with open(claimid+"/defect.json", "w") as json_file:
            json.dump(defect_dict, json_file)
        print("Time Taken for defect detection > ",time.time()-start)
        
    def tyre_mask_thread():
        # To detect tyre and mask accurately for splitted images
        tyre_mask_func(claimid)
        
    start3 = time.time()
    t1 = threading.Thread(target=defect_thread)
    t1.start()
    t1.join()
    defect_flag = "false"
    if "Sidewall Cut" in finaldefect or "Shoulder Cut" in finaldefect:
        t2 = threading.Thread(target=tyre_mask_thread)
        t2.start()

        t2.join()
        print("Time Taken for Tyre Masking > ",time.time()-start3)
    
        start2 = time.time()
        # check if the defect detected is on the tire
        defect_flag = tyre_mask_defect(claimid)  
        print("Time Taken for Checking defect on tyre (verification) > ",time.time()-start2)
    
    # For merging the image
    try:
        merge_image(claimid)
    except Exception as e:
        print(e)
    try:
        directory = claimid + '/Splitted_Images'
        shutil.rmtree(directory)
    except Exception as e:
        print("Exception occured in removing directories >> ",e)
        
    if len(finaldefect) > 0:
        defect_cls2 = ",".join(finaldefect)
    
    if defect_flag=="true":
        if defect_cls2=="" or defect_cls2==None or defect_cls2==",":
            return_defect = "Good"
            return_code = 271
        else:
            if ("Sidewall Cut" in defect_cls2) or ("Shoulder Cut" in defect_cls2):
                return_defect = "Bad"
                return_code = 261
                print("defect res>>>",defect_cls2)
            else:
                return_defect = "Good"
                return_code = 271
    else:
        return_defect = "Good"
        return_code = 271
    
    return return_defect,return_code