import pymysql
from datetime import datetime

def write_txt(claim_id,text):
    with open(claim_id+'/txt_logs.txt', 'a') as f:
        f.write(str(text))