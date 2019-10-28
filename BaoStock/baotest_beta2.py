# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:22:32 2019

@author: holya
"""

import baostock as bs
import pandas as pd

lg = bs.login()

print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

rs = bs.query_deposit_rate_data(start_date="2015-01-01", end_date="2018-12-31")
print('query_deposit_rate_data respond error_code:'+rs.error_code)
print('query_deposit_rate_data respond  error_msg:'+rs.error_msg)

data_list = []
while (rs.error_code == '0') & rs.next():
    
    data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)

bs.logout()
