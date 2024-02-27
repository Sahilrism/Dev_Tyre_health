import codecs,json,cv2,os
import numpy as np
from functools import reduce
from math import *

def percentage_comp(x,y):
    perc = 0
    axis = ""
    print("y    x")
    print( y, " " ,x )
    if x>y:
        axis = "y"
        perc = 100 - (y/x)*100
    else:
        axis = "x"
        perc = 100 - (x/y)*100
    print("*"*50)
    print("compressed axis >> ",axis)
    print("percentage compressed >> ",perc)
    print("*"*50)
    return perc,axis

def coin_calc(claimid,fdict):

    # load predicted mask
    file_path = claimid +"/mask_coord.json"
    obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
    b_new = json.loads(obj_text)
    mask_array = b_new
    mask_array = np.array(mask_array)   ###changes for numpy array
    print(len(mask_array))

    # minn = 100000
    # maxx = 0
    minimum_y_list = []

    min_y = mask_array[np.argmin(mask_array[:, 1])]
    print("min_y $$$$$$$$ ",min_y)

    minimum_y_list = mask_array[mask_array[:, 1] == min_y[1]]
    # print("minimum_y_list >> ",minimum_y_list)
    
    print("len of minimum_y_list >> ",len(minimum_y_list))
    if len(minimum_y_list)%2==0:
        min_y = minimum_y_list[int(len(minimum_y_list)/2)]
    else:
        min_y = minimum_y_list[int((len(minimum_y_list)-1)/2)]
    # print("min_y >> ",min_y)

    # minn = 100000
    # maxx = 0
    
    max_x = mask_array[np.argmax(mask_array[:, 0])]
    print("max_x $$$$$$$$$ ",max_x)

    # minn = 100000
    # maxx = 0

    min_x = mask_array[np.argmin(mask_array[:, 0])]

    print("min_x >> ",min_x)
    
    fingerflag = False
    # calculate finger position using predicted bounding box
    
    #########################################################
    coordinates = mask_array
    # Calculate the centroid of the detected mask
    x_coordinates = [coord[0] for coord in coordinates]
    y_coordinates = [coord[1] for coord in coordinates]
    # print("CC1",x_coordinates)
    # print("CC2",y_coordinates)
    centroid_x = np.mean(x_coordinates)
    centroid_y = np.mean(y_coordinates)
    print("CC3",centroid_x)
    print("CC4",centroid_y)

    # Determine the orientation of the coin
    if centroid_x < min(x_coordinates):
        orientation = "Coin is held from the right side"
    elif centroid_x > max(x_coordinates):
        orientation = "Coin is held from the left side"
    elif centroid_y > max(y_coordinates):
        orientation = "Coin is held from the top portion"
    else:
        orientation = "Coin orientation is unclear"

    print("orientation ============ ",orientation)
    #################################################################
    
    # to check if finger is held from top portion
    if not(fingerflag):
        xmaxcord = max_x
        xmincord = min_x
        # for i in mask_array:
        #     if max_x[1]==i[1] and i[0]<max_x[0]:
        #         xmincord = i
        distx = abs(xmaxcord[0] - xmincord[0])
        print("xmaxcord ********** ",xmaxcord)
        print("xmincord ********** ",xmincord)
        print("distx ********** ",distx)
        
        ymincord = min_y
        for i in mask_array:
            if min_y[0]==i[0] and i[1]>min_y[1]:
                ymaxcord = i
        disty = abs(ymaxcord[1] - ymincord[1])
        print("ymaxcord ********** ",ymaxcord)
        print("ymincord ********** ",ymincord)
        print("disty ********** ",disty)

        
    # to check if finger is held from left or right portion
    coinbox = fdict["10"]
    ax, wh = coinbox
    try:
        del fdict["10"]
    except Exception as e:
        print(e)
    midwidth = ceil((ax[0]+wh[0])/2)
    if len(fdict)>0:
        print(fdict)
        bbox = list(fdict.values())[0]
        # print("detcted finger bbox >> ",bbox)
        w = bbox[1][0]
        x = bbox[0][0]
        midw = ceil((w+x)/2)
        print("midwidth midw >>"  , midwidth," ",midw)
        if midw>midwidth:
            print("finger is on right side(detected)")
            finger_pos = "right"
        else:
            print("finger is on left side(detected)")
            finger_pos = "left"
    else:
        if abs(min_y[0]-max_x[0]) > abs(min_y[0]-min_x[0]):
            print("finger is on left side")
            finger_pos = "left"
        else:
            print("finger is on right side")
            finger_pos = "right"

    x_coord_list = []
    
    if finger_pos == "right":
        print("yessssssssssssssssssss")
        x_coord_list = mask_array[min_x[0] == mask_array[:, 0]]
    else:
        print("noooooooooooooo")
        x_coord_list = mask_array[max_x[0] == mask_array[:, 0]]
    # print("x_coord_list >>> ",x_coord_list)
    print("len of x_coord_list >>> ",len(x_coord_list))

    if len(x_coord_list)%2==0:
        x_calculate = x_coord_list[int(len(x_coord_list)/2)]
    else:
        x_calculate = x_coord_list[int((len(x_coord_list)-1)/2)]
        
    print("x to calculate >>>> ",x_calculate)
    print("y to calculate >>>> ",min_y)
    centre_cord = [min_y[0],x_calculate[1]]
    print("centre co-ordinate >>>> ",centre_cord)
    
    # to find compresion and percenatge compressed
    y_ = abs(centre_cord[1]-min_y[1])
    x_ = abs(centre_cord[0]-x_calculate[0])
    perc_,axis_ = percentage_comp(x_,y_)
    
    # if axis_=="y":
    #     center_pixel = abs(x_calculate[0] - centre_cord[0])
    # else:
    #     center_pixel = abs(min_y[1] - centre_cord[1])
    center_pixel = abs(min_y[1] - centre_cord[1])
        
    print("pixels in between centre and calculated coordinate (not compressed) >>>> ",center_pixel)

    mm_pix = center_pixel/13.5
    print("mm_pix >>> ",mm_pix)

    for i in mask_array:
        if min_y[0]==i[0] and i[1]>min_y[1]:
            down_max_y = i

    print("down_max_y >> ",down_max_y)

    total_pix = abs(down_max_y[1] - min_y[1])
    print("total_pix >> ",total_pix)

    read = abs(total_pix/mm_pix)
    print("coin >> ",read)

    final_read = abs(27 - read)
    print("read >> ",final_read)
    final = final_read
    
    # calculate final reading after compression percentage calculation
    if perc_>0:
        # considering 0.3 buffer for 1% compression at x-axis and y-axis as analysed on multiple cases
        if axis_=="x":
            buff = 0.3
        else:
            buff = 0.15
        buffcalc = buff * perc_
        print("Buffer to be added : ",buffcalc)
        # final += buffcalc
    final += 0.30
    print("final read >> ",final)
        
        
    
    # Rounding off final Reading
    rounded_num = round(final + 0.1, 1)
    # rounded_num += 0.2

    if "." in str(rounded_num):
        final_reading = rounded_num
    else:
        final_reading = 0
        
    return final_reading
