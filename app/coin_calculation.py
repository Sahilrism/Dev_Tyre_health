# import codecs,json,cv2,os
# import numpy as np
# from functools import reduce
# from math import *

# def percentage_comp(x,y):
#     perc = 0
#     axis = ""
#     print("y    x")
#     print( y, " " ,x )
#     if x>y:
#         axis = "y"
#         perc = 100 - (y/x)*100
#     else:
#         axis = "x"
#         perc = 100 - (x/y)*100
#     print("*"*50)
#     print("compressed axis >> ",axis)
#     print("percentage compressed >> ",perc)
#     print("*"*50)
#     return perc,axis

# def coin_calc(claimid,fdict):

#     # load predicted mask
#     file_path = claimid +"/mask_coord.json"
#     obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
#     b_new = json.loads(obj_text)
#     mask_array = b_new
#     mask_array = np.array(mask_array)   ###changes for numpy array
#     print(len(mask_array))

#     # minn = 100000
#     # maxx = 0
#     minimum_y_list = []

#     min_y = mask_array[np.argmin(mask_array[:, 1])]

#     minimum_y_list = mask_array[mask_array[:, 1] == min_y[1]]
#     # print("minimum_y_list >> ",minimum_y_list)
    
#     print("len of minimum_y_list >> ",len(minimum_y_list))
#     if len(minimum_y_list)%2==0:
#         min_y = minimum_y_list[int(len(minimum_y_list)/2)]
#     else:
#         min_y = minimum_y_list[int((len(minimum_y_list)-1)/2)]
#     # print("min_y >> ",min_y)

#     # minn = 100000
#     # maxx = 0
    
#     max_x = mask_array[np.argmax(mask_array[:, 0])]
#     print("max_x >> ",max_x)

#     # minn = 100000
#     # maxx = 0

#     min_x = mask_array[np.argmin(mask_array[:, 0])]

#     print("min_x >> ",min_x)
    
#     # calculate finger position using predicted bounding box
#     coinbox = fdict["10"]
#     ax, wh = coinbox
#     try:
#         del fdict["10"]
#     except Exception as e:
#         print(e)
#     midwidth = ceil((ax[0]+wh[0])/2)
#     if len(fdict)>0:
#         print(fdict)
#         bbox = list(fdict.values())[0]
#         # print("detcted finger bbox >> ",bbox)
#         w = bbox[1][0]
#         x = bbox[0][0]
#         midw = ceil((w+x)/2)
#         print("midwidth midw >>"  , midwidth," ",midw)
#         if midw>midwidth:
#             print("finger is on right side(detected)")
#             finger_pos = "right"
#         else:
#             print("finger is on left side(detected)")
#             finger_pos = "left"
#     else:
#         if abs(min_y[0]-max_x[0]) > abs(min_y[0]-min_x[0]):
#             print("finger is on left side")
#             finger_pos = "left"
#         else:
#             print("finger is on right side")
#             finger_pos = "right"

#     x_coord_list = []
#     if finger_pos == "right":
#         x_coord_list = mask_array[min_x[0] == mask_array[:, 0]]
#     else:
#         x_coord_list = mask_array[max_x[0] == mask_array[:, 0]]
#     # print("x_coord_list >>> ",x_coord_list)
#     print("len of x_coord_list >>> ",len(x_coord_list))

#     if len(x_coord_list)%2==0:
#         x_calculate = x_coord_list[int(len(x_coord_list)/2)]
#     else:
#         x_calculate = x_coord_list[int((len(x_coord_list)-1)/2)]
        
#     print("x to calculate >>>> ",x_calculate)
#     print("y to calculate >>>> ",min_y)
#     centre_cord = [min_y[0],x_calculate[1]]
#     print("centre co-ordinate >>>> ",centre_cord)
    
#     # to find compresion and percenatge compressed
#     y_ = abs(centre_cord[1]-min_y[1])
#     x_ = abs(centre_cord[0]-x_calculate[0])
#     perc_,axis_ = percentage_comp(x_,y_)
    
#     # if axis_=="y":
#     #     center_pixel = abs(x_calculate[0] - centre_cord[0])
#     # else:
#     #     center_pixel = abs(min_y[1] - centre_cord[1])
#     center_pixel = abs(min_y[1] - centre_cord[1])
        
#     print("pixels in between centre and calculated coordinate (not compressed) >>>> ",center_pixel)

#     mm_pix = center_pixel/13.5
#     print("mm_pix >>> ",mm_pix)

#     for i in mask_array:
#         if min_y[0]==i[0] and i[1]>min_y[1]:
#             down_max_y = i

#     print("down_max_y >> ",down_max_y)

#     total_pix = abs(down_max_y[1] - min_y[1])
#     print("total_pix >> ",total_pix)

#     read = abs(total_pix/mm_pix)
#     print("coin >> ",read)

#     final_read = abs(27 - read)
#     print("read >> ",final_read)
#     final = final_read
    
#     # calculate final reading after compression percentage calculation
#     if perc_>0:
#         # considering 0.3 buffer for 1% compression at x-axis and y-axis as analysed on multiple cases
#         if axis_=="x":
#             buff = 0.3
#         else:
#             buff = 0.15
#         buffcalc = buff * perc_
#         print("Buffer to be added : ",buffcalc)
#         # final += buffcalc
#     final += 0.30
#     print("final read >> ",final)
        
        
    
#     # Rounding off final Reading
#     rounded_num = round(final + 0.1, 1)
#     # rounded_num += 0.2

#     if "." in str(rounded_num):
#         final_reading = rounded_num
#     else:
#         final_reading = 0
        
#     return final_reading
#################new codeeeeeeeeeeeeeeeee###########

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

def coin_calc(claimid,fdict,hand_loc,finger_det,coin_size_full):

    print('Hand Location >>>>>>>',hand_loc,finger_det)
    # load predicted mask
    file_path = claimid +"/mask_coord.json"
    obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
    b_new = json.loads(obj_text)
    mask_array = b_new
    mask_array = np.array(mask_array)   ###changes for numpy array
    print(len(mask_array))
    # print("mask|"*25)
    # print(mask_array)
    # print("mask|"*25)

    # minn = 100000
    # maxx = 0
    minimum_y_list = []

    min_y = mask_array[np.argmin(mask_array[:, 1])]

    minimum_y_list = mask_array[mask_array[:, 1] == min_y[1]]
    print("*"*50)
    print(min_y)
    print("*"*50)
    # print(minimum_y_list)
    # print("minimum_y_list >> ",minimum_y_list)

    ##########################---------- Minimum X AND Minimum Y----------- ########################################
    
    print("Total minimum_y_list >> ",len(minimum_y_list))
    if len(minimum_y_list)%2==0:
        mini_y = minimum_y_list[int(len(minimum_y_list)/2)]
    else:
        mini_y = minimum_y_list[int((len(minimum_y_list)-1)/2)]

    print("Minimum y Center >> ",mini_y)


    # minn = 100000
    # maxx = 0
    
    max_x = mask_array[np.argmax(mask_array[:, 0])]
    maximum_x_list = mask_array[mask_array[:, 0] == max_x[0]]
    print("*"*25)
    print("total maximum x >>>",len(maximum_x_list))
    # print(maximum_x_list)

    min_x = mask_array[np.argmin(mask_array[:, 0])]
    minimum_x_list = mask_array[mask_array[:, 0] == min_x[0]]
    print("*"*25)
    print("total minimum x >>>",len(minimum_x_list))
    # print(minimum_x_list)

    if len(maximum_x_list)%2==0:
        maxi_x = maximum_x_list[int(len(maximum_x_list)/2)]
    else:
        maxi_x = maximum_x_list[int((len(maximum_x_list)-1)/2)]

    print("Maximum x >>>>>", maxi_x)        

    if len(minimum_x_list)%2==0:
        mini_x = minimum_x_list[int(len(minimum_x_list)/2)]
    else:
        mini_x = minimum_x_list[int((len(minimum_x_list)-1)/2)]

    print("Minimum x >>>>> ",mini_x)

    x_center = ceil((maxi_x[0] + mini_x[0])/2)
    y_center = ceil((maxi_x[1] + mini_x[1])/2)

##########################---------- Minimum X AND Minimum Y   (END ) ----------- ########################################


##############################    CAMERA POSITION   ################################
    camera_pos = "Front"
    if mini_x[1] > maxi_x[1]:
        camera_y_diff = abs(mini_x[1] - maxi_x[1])
        if camera_y_diff > 7 :
            camera_pos = "Left"
        else:
            camera_pos = 'Front'

    elif min_x[1] < maxi_x[1] :
        camera_y_diff = abs(mini_x[1] - maxi_x[1])
        if camera_y_diff > 10 :
            camera_pos = "Right"
        else:
            camera_pos = 'Front'

##############################   CAMERA POSITION END  ##################################


##############################    HAND POSITION   ##############################

    if hand_loc == 'Upper':
        print("Finger BOX >>>>>>", fdict['finger'])
        print("Coin upper BOX >>>>>", fdict['10_upper'])

        finger_box_center = (fdict['finger'][0][0] + fdict['finger'][1][0])/2
        coin_box_center =  (fdict['10_upper'][0][0] + fdict['10_upper'][1][0])/2

        if finger_box_center < coin_box_center:
            finger_hold = "Left"
            print("Holding Hand >>>>>", finger_hold)
        
        elif finger_box_center > coin_box_center:
            finger_hold = "Right"
            print("Holding Hand >>>>>", finger_hold)

##############################    HAND POSITION END   ##############################
    
    # calculate finger position using predicted bounding box
    if "10" in fdict.keys():
        coinbox = fdict["10"]
        ax, wh = coinbox
        print('*'*50)
        print(ax)
        print(wh)
        try:
            del fdict["10"]
        except Exception as e:
            print(e)
        midwidth = ceil((ax[0]+wh[0])/2)
        print('*'*50)
        print('Midwidth >>>>>',midwidth)
    elif "5" in fdict.keys():
        coinbox = fdict["5"]
        ax, wh = coinbox
        print('*'*50)
        print(ax)
        print(wh)
        try:
            del fdict["5"]
        except Exception as e:
            print(e)
        midwidth = ceil((ax[0]+wh[0])/2)
        print('*'*50)
        print("Midwidth >>>>>",midwidth)
    if hand_loc == "Upper":
        print("Yes inside Upper")
        # center_coin = floor((max_x[0] - min_x[0]) / 2)
        center_coin = [x_center,y_center]
        print("Upper center coin >>>>>>>>>>",center_coin)
        finger_pos = "upper"

    elif hand_loc == "Not Detected":
        print("Yes inside Upper")
        # center_coin = floor((max_x[0] - min_x[0]) / 2)
        center_coin = [x_center,y_center]
        print("Upper center coin >>>>>>>>>>",center_coin)
        finger_pos = "upper"

    elif len(fdict)>0 and hand_loc == "Normal":
        # print(fdict)
        if "left_finger" in fdict.keys():
            bbox = fdict["left_finger"]

        elif "right_finger" in fdict.keys():
            bbox = fdict["right_finger"]
        # bbox = list(fdict.values())[0]
        print("*"*50)
        # print(bbox)
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
        if abs(mini_y[0]-max_x[0]) > abs(mini_y[0]-min_x[0]):
            print("finger is on left side")
            finger_pos = "left"
        else:
            print("finger is on right side")
            finger_pos = "right"

    x_coord_list = []
    coin_upper_center = []
    if finger_pos == "upper":
        coin_upper_center.append(center_coin)
    elif finger_pos == "right":
        x_coord_list = mask_array[min_x[0] == mask_array[:, 0]]
    else:
        x_coord_list = mask_array[max_x[0] == mask_array[:, 0]]
    # print("x_coord_list >>> ",x_coord_list)
    # try:
    #     print("len of x_coord_list >>> ",len(x_coord_list))
    # except Exception as e:
    #     print("Not working sayeed logic")

    #Coin size from db and divide it in half for calculation

    coin_size_mm = float(coin_size_full) / 2
    print("*"*50)
    print(f"Coin size from db is : {coin_size_full} and after dividing it in half it is : {coin_size_mm} ")
    print("*"*50)

    if len(coin_upper_center)>0:
        print("New Upper hand Logic Start....")
        for i in mask_array:
            if center_coin[0]==i[0] and i[1]>min_y[1]:
                down_max_y = i
        print("Down max Y >>>>>>", down_max_y)   
        
        center_diff = abs(center_coin[0] - maxi_x[0])
        per_pixel = center_diff / coin_size_mm
        print("PERRRRRR PIXXXX >>>>>>", per_pixel)

        # down_y_diff = abs(down_max_y[1] - abs(min_x[0] - max_x[0]))
        down_center_diff = abs(center_coin[1] - down_max_y[1])
        down_y_diff = abs(center_diff - down_center_diff)
        print("DOWN DIFFF>>>>>",down_y_diff)

        predict_max_y = down_max_y[1] + down_y_diff
        print("PREDICT Y>>>>>>>", predict_max_y)

        raw_read = down_y_diff / per_pixel
        print("RAW READING >>>>>",raw_read)

        print("Camera Position >>>>>", camera_pos)
        print("Camera Y difference >>>>",camera_y_diff)

        if hand_loc == "Upper":
            if camera_pos == "Left" and finger_hold == "Left":
                buff1 = 0.1
                final = raw_read + buff1
                print("Buffer Added >>>>>", buff1)
                # if camera_y_diff >10:
                #     buff1 = camera_y_diff/2
                #     buff2 = buff1 / 10
                #     final = raw_read + buff2
                #     print("Buffer Added >>>>>", buff2)
                
                # else:
                #     buff1 = camera_y_diff/10
                #     final = raw_read + buff1
                #     print("Buffer Added >>>>>", buff1)

            elif camera_pos == "Left" and finger_hold == "Right":
                if camera_y_diff >=8 and camera_y_diff <=14:
                    buff1 = camera_y_diff/2
                    buff2 = buff1 / 10
                    final = raw_read + buff2
                    print("Buffer Added >>>>>", buff2)

                elif camera_y_diff >14:
                    buff1 = camera_y_diff/3
                    buff2 = buff1 / 10
                    final = raw_read + buff2
                    print("Buffer Added >>>>>", buff2)
                
                else:
                    buff1 = camera_y_diff/10
                    final = raw_read + buff1
                    print("Buffer Added >>>>>", buff1)

            elif camera_pos == "Front":
                if camera_y_diff >10:
                    buff1 = camera_y_diff/2
                    buff2 = buff1 / 10
                    final = raw_read + 0.3
                    print("Buffer Added >>>>>", buff2)
                
                else:
                    buff1 = camera_y_diff/10
                    final = raw_read + 0.3
                    print("Buffer Added >>>>>", buff1)

            elif camera_pos == "Right" and finger_hold == "Left":
                if camera_y_diff >10 and camera_y_diff<=14:
                    buff1 = camera_y_diff/2
                    buff2 = buff1 / 10
                    # final = raw_read - 0.8
                    final = raw_read + buff2
                    print("Buffer Added >>>>>", buff2)

                elif camera_y_diff>14:
                    buff1 = camera_y_diff/3
                    buff2 = buff1 / 10
                    # final = raw_read - 0.8
                    final = raw_read + buff2
                    print("Buffer Added >>>>>", buff2)

            elif camera_pos == "Right" and finger_hold == "Right":
                if camera_y_diff >=8 and camera_y_diff <=14:
                    buff1 = camera_y_diff/2
                    buff2 = buff1 / 10
                    # final = raw_read - 0.8
                    final = raw_read + buff2
                    print("Buffer Added >>>>>", buff2)

                elif camera_y_diff>14:
                    buff1 = camera_y_diff/3
                    buff2 = buff1 / 10
                    # final = raw_read - 0.8
                    final = raw_read + buff2
                    print("Buffer Added >>>>>", buff2)

            
                
                # else:
                #     buff1 = camera_y_diff/10
                #     final = raw_read - buff1

        else:
            final = raw_read

        # final_reading = round(final,1) 
        final_reading = round(final,1)
        print("FINAL READING >>>>>>", final_reading)
        print("HAND LOCATION >>>>>",hand_loc)
        minin_x =mini_x.tolist()
        maxin_x =maxi_x.tolist()
        calculated_data ={
                        "Coin_center":center_coin,
                        "Minimum_x" : minin_x,
                        "Maximum_x":maxin_x,
                        "Hand_location": hand_loc,
                        "Finger_Detection": finger_det,
                        "Camera_Position":camera_pos,
                        "Raw_read":raw_read,
                        "Final_read":final_reading
                        }

        

    else:
        if len(x_coord_list)%2==0:
            x_calculate = x_coord_list[int(len(x_coord_list)/2)]
        else:
            x_calculate = x_coord_list[int((len(x_coord_list)-1)/2)]
            
        print("X to calculate >>>> ",x_calculate)
        print("Y to calculate >>>> ",mini_y)
        centre_cord = [mini_y[0],x_calculate[1]]
        print("Centre Co-ordinate >>>> ",centre_cord)
        
        # to find compresion and percenatge compressed
        y_ = abs(centre_cord[1]-mini_y[1])
        x_ = abs(centre_cord[0]-x_calculate[0])
        perc_,axis_ = percentage_comp(x_,y_)
        
        # if axis_=="y":
        #     center_pixel = abs(x_calculate[0] - centre_cord[0])
        # else:
        #     center_pixel = abs(min_y[1] - centre_cord[1])
        center_pixel = abs(mini_y[1] - centre_cord[1])
            
        print("Pixels in between centre and calculated coordinate (not compressed) >>>> ",center_pixel)

        mm_pix = center_pixel/coin_size_mm
        print("Pixel per mm >>>>> ",mm_pix)

        for i in mask_array:
            if mini_y[0]==i[0] and i[1]>mini_y[1]:
                down_max_y = i

        print("Down_max_y >>>>> ",down_max_y)

        total_pix = abs(down_max_y[1] - mini_y[1])
        print("Total_pix >>>>> ",total_pix)

        read = abs(total_pix/mm_pix)
        print("Coin >> ",read)

        final_read = abs(float(coin_size_full) - read)
        print("Read >> ",final_read)
        final = final_read
        
        # calculate final reading after compression percentage calculation
        if perc_>0:
            # considering 0.3 buffer for 1% compression at x-axis and y-axis as analysed on multiple cases
            if axis_=="x":
                buff = 0.3
            else:
                buff = 0.15
            buffcalc = buff * perc_
            print("Buffer need to add : ",buffcalc)
            # final += buffcalc
        final += 0.30

        #######################------------ SAHIL BUFFER --------------#######################
        
        # predict_center = floor((centre_cord[0] + centre_cord[1])/2)
        # print("PREDICT CENTER >>>>>", predict_center)



        # x_diff = abs(predict_center - max_x[0])
        # print("X_BUFF_DIFF >>>", x_diff) 

        # y_diff = abs(predict_center - min_y[1])
        # print("Y_BUFF_DIFF >>>", y_diff)

        # # center_diff = abs(centre_cord[0] - centre_cord[1])

        # permm = (x_diff + y_diff)/27
        # print("PerMMMMMMMMMM >>>>>",permm)

        # diff_center = abs(centre_cord[0] - centre_cord[1])
        # print("CENTER DIFFERENT >>>>>>>",diff_center)

        # if diff_center>25 and diff_center <30:
        #     buff_1 = diff_center /2
        #     buff_2 = buff_1/permm
        #     buff_final = final - buff_2
        #     print("BUFFER ADDED >>>>>", buff_2)
        #     modified_read = buff_final

        # elif diff_center>=30 and diff_center <40:
        #     buff_1 = diff_center /3
        #     buff_2 = buff_1/permm
        #     buff_final = final - buff_2
        #     print("BUFFER ADDED >>>>>", buff_2)
        #     modified_read = buff_final

        
        # elif diff_center>=40 and diff_center <50:
        #     buff_1 = diff_center /4
        #     buff_2 = buff_1/permm
        #     buff_final = final - buff_2
        #     print("BUFFER ADDED >>>>>", buff_2)
        #     modified_read = buff_final
        # else:
        #     # buff_1 = abs(x_diff - y_diff)
        #     buff_2 = diff_center /4
        #     buff_3 = buff_2/permm
        #     buff_4 = buff_3*4

        #     if y_diff < x_diff:
        #         buff_final = final - buff_4
        #         print("BUFFER SUBTRACTED >>>>>", buff_4)

        #     elif y_diff > x_diff:
        #         buff_final = final + buff_4
        #         print("BUFFER ADDED >>>>>", buff_4)

        #     modified_read = buff_final


        # final = round(modified_read,1)

        print("final read >> ",final)
            
            
        
        # Rounding off final Reading
        rounded_num = round(final + 0.1, 1)

        # rounded_num += 0.2

        if "." in str(rounded_num):
            final_reading = rounded_num
        else:
            final_reading = 0
        
        x_to_calculate = x_calculate.tolist()
        down_maxi_y = down_max_y.tolist()
        minin_y = mini_y.tolist()
        

        calculated_data = {
                        "Coin_center":str(centre_cord),
                        "X" : str(x_to_calculate),
                        "Y":str(minin_y),
                        "Down Y": str(down_maxi_y),
                        "Pixel per mm": str(mm_pix),
                        "Hand_location": hand_loc,
                        "Finger_Detection": finger_det,
                        "Raw_read":str(final_read),
                        "Final_read":str(final_reading),
                        "Logic":"old"
                        }
        
    return final_reading,calculated_data
