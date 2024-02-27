import re


def number_regerx(number):

    ret_num=""
    reg_flag = "false"

    try:

        x=re.findall(".*?([A-Za-z]{2})(\d{2})([A-Za-z]{2})(\d{4}).*?",number.upper())[0]
        print("MH43BU9181 regex working")
        delim=""
        x1= ''.join([str(ele) + delim for ele in x])
        print("x1==========",x1)
        ret_num=x1
        reg_flag = "true"

    except Exception as e:
        try:
            x=re.findall(".*?([A-Za-z]{2})(\d{2})([A-Za-z]{1})(\d{4}).*?",number.upper())[0]
            print("MH43U9181 regex working")
            delim=""
            x1= ''.join([str(ele) + delim for ele in x])
            print("x1==========",x1)
            ret_num=x1
            reg_flag = "true"


        except Exception as e:
            try:
                x=re.findall(".*?(\d{2})([A-Za-z]{2})(\d{4})([A-Za-z]{2}).*?",number.upper())[0]
                print("22BH9181BU regex working")
                delim=""
                x1= ''.join([str(ele) + delim for ele in x])
                ret_num=x1
                print("x1==========",x1)
                reg_flag = "true"
            except Exception as e:
                try:
                    x=re.findall(".*?(\d{2})([A-Za-z]{2})(\d{4})([A-Za-z]{1}).*?",number.upper())[0]
                    print("22BH9181U regex working")
                    delim=""
                    x1= ''.join([str(ele) + delim for ele in x])
                    ret_num=x1
                    print("x1==========",ret_num)
                    reg_flag = "true"
                    
                except Exception as e:
                    # print("Exception : ",e)
                    try:    
                        x=re.findall(".*?([A-Za-z]{2})(\d{1})([A-Za-z]{1})(\d{4}).*?",number.upper())[0]
                        print("DL1T2348 regex working")
                        delim=""
                        x1= ''.join([str(ele) + delim for ele in x])
                        ret_num=x1
                        print("x1==========",x1)
                        reg_flag = "true"
                    except Exception as e:
                        # print("Exception : ",e)
                        try:    
                            x=re.findall(".*?([A-Za-z]{2})(\d{2})([A-Za-z]{2})(\d{3}).*?",number.upper())[0]
                            print("MH03TA234 regex working")
                            delim=""
                            x1= ''.join([str(ele) + delim for ele in x])
                            ret_num=x1
                            print("x1==========",x1)
                            reg_flag = "true"
                        except Exception as e:
                        # print("Exception : ",e)
                            try:    
                                x=re.findall(".*?([A-Za-z]{2})(\d{2})([A-Za-z]{2})(\d{3})([A-Za-z]{1}).*?",number.upper())[0]
                                print("MH02AB420X regex working")
                                delim=""
                                x1= ''.join([str(ele) + delim for ele in x])
                                ret_num=x1
                                print("x1==========",x1)
                                reg_flag = "true"
                            except Exception as e:
                                try:
                                    x=re.findall(".*?([A-Za-z]{2})(\d{1})([A-Za-z]{3})(\d{4}).*?",number.upper())[0]
                                    print("DL1TAA2348 regex working")
                                    delim=""
                                    x1= ''.join([str(ele) + delim for ele in x])
                                    ret_num=x1
                                    print("x1==========",x1)
                                    reg_flag = "true"
                                except Exception as e:
                                    print("Exception : ",e)
         
        if reg_flag == "false":                  
            try:    
                x=re.findall(".*?([A-Za-z]{2})(\d{6})([A-Za-z]{2}).*?",number.upper())[0]
                print("fancy regex working")
                delim=""
                x1= ''.join([str(ele) + delim for ele in x])
                ret_num=x1
                lastchar = ret_num[-2:]
                # MH042227GU
                initchar = ret_num.replace(lastchar,"")
                fourdig = initchar[-4:]
                start_char = initchar.replace(fourdig,"")
                return_number = start_char + lastchar + fourdig
                print("Modified num : ",return_number)
                try:    
                    x=re.findall(".*?([A-Za-z]{2})(\d{2})([A-Za-z]{2})(\d{4}).*?",return_number.upper())[0]
                    print("MH43BU9181 regex working")
                    delim=""
                    x1= ''.join([str(ele) + delim for ele in x])
                    ret_num = x1
                    reg_flag = "true"
                except Exception as e:
                    ret_num = return_number
            except Exception as e:
                try:
                    if reg_flag == "false":                     
                        x=re.findall(".*?([A-Za-z]{2})(\d{6})([A-Za-z]{1}).*?",number.upper())[0]
                        print("fancy regex working")
                        delim=""
                        x1= ''.join([str(ele) + delim for ele in x])
                        ret_num=x1
                        lastchar = ret_num[-1:]
                        # MH042227GU
                        initchar = ret_num.replace(lastchar,"")
                        fourdig = initchar[-4:]
                        start_char = initchar.replace(fourdig,"")
                        return_number = start_char + lastchar + fourdig
                        print("Modified num : ",return_number)
                        try:    
                            x=re.findall(".*?([A-Za-z]{2})(\d{2})([A-Za-z]{1})(\d{4}).*?",return_number.upper())[0]
                            print("MH43BU9181 regex working")
                            delim=""
                            x1= ''.join([str(ele) + delim for ele in x])
                            ret_num = x1
                            reg_flag = "true"
                        except Exception as e:
                            ret_num = return_number
                except Exception as e:
                    # print("Exception : ",e)
                    try:
                        x=re.findall(".*?([A-Za-z]{2})(\d{2})([A-Za-z]{2})(\d{1}).*?",number.upper())[0]
                        print("MH 43 BU 9 regex working")
                        delim=""
                        x1= ''.join([str(ele) + delim for ele in x])
                        print("x1==========",x1)
                        ret_num=x1
                        reg_flag = "true"
                    except Exception as e:
                        print("Exception : ",e)
                        
    return ret_num,reg_flag