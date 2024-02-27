import os,glob,cv2,json

def load_json(json_path):
    f = open(json_path)
    data = json.load(f)
    return data

def tyre_mask_defect(claimid):
    defect_path = claimid + "/defect.json"
    tyre_mask_path = claimid + "/tyre_mask.json"
    
    defect_data = load_json(defect_path)
    tyre_mask_data = load_json(tyre_mask_path)
    
    list1 = defect_data.keys()
    list2 = tyre_mask_data.keys()
    print(list1)
    print(list2)
    common_ele = set(list1).intersection(list2)
    common_list = list(common_ele)
    print(common_list)
    damage_flag = "false"
    try:
        for i in common_list:
            defect_common_value = defect_data[i]
            tyre_mask_common_value = tyre_mask_data[i]
            print(len(defect_common_value))
            print(len(tyre_mask_common_value))
            try:
                indices_of_ones_in_defect = [i for i, val in enumerate(defect_common_value) if val == 1]
                # print(indices_of_ones_in_defect)
                all_ones_in_tyre_mask = all(tyre_mask_common_value[i] == 1 for i in indices_of_ones_in_defect)
                print(all_ones_in_tyre_mask)
                if all_ones_in_tyre_mask:
                    damage_flag = "true"
                else:
                    damage_flag = "false"
            except Exception as e:
                damage_flag = "false"
    except Exception as e:
        damage_flag = "false"
    return damage_flag

    
    
# tyre_mask_defect("/home/checkexplore/Desktop/Tyre_Health/Claim_ID/a1de9424-dc6a-4fe6-9787-19e26724479a_defect-outside")
