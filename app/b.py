import pymysql


def coin_size():
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



def get_coin_size_by_id(data_tuple, target_id):
    for inner_tuple in data_tuple:
        if inner_tuple[0] == target_id:
            return inner_tuple
    return None  # ID not mentioned in db

data = coin_size()
print(data)
target_id = 2
result = get_coin_size_by_id(data, target_id)

# print(f"Inner tuple with ID {target_id}: {result}")

# if result != None:
#     coin_name = result[1]
#     coin_size_mm = result[2]
#     print(f"coin_name: {coin_name}, coin_size: {coin_size_mm}")
# else:
#     print("Provide proper ID")

if (result[1]=="10_rs") and (result[3]=="2016-2019"):
    print("Yess")

else:
    print("No")