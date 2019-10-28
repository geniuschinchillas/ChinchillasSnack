# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:13:43 2019

@author: holya
"""

import baostock as bs
import pandas as pd

lg = bs.login()

print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

rs = bs.query_history_k_data_plus("sz.002098",
    "date,code,open,high,low,close,volume,amount,adjustflag",
    start_date='2014-01-01', end_date='2018-12-31',
    frequency="30", adjustflag="3")
print('query_history_k_data_plus respond error_code:'+rs.error_code)
print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)

data_list = []
while (rs.error_code == '0') & rs.next():
    data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)

bs.logout()
