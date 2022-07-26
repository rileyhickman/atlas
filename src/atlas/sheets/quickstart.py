#!/usr/bin/env python

import gspread
import pandas as pd

sa = gspread.service_account()

sh = sa.open('test_atlas')

wks = sh.worksheet('Sheet1')

print('rows : ', wks.row_count)
print('cols : ', wks.col_count)


print(wks.acell('A1'))
print(wks.acell('A1').value)

print(wks.cell(3, 4))
print(wks.cell(3, 4).value)


print(wks.get('A1:C3')) # list of lists for each row

# wks.update('A1', 'param0') # update the value of a single cell

data_dict = wks.get_all_records()

df = pd.DataFrame(data_dict)
print(df.shape)
print(df.columns)
print(df.head())

# update a worksheet with a pandas dataframe
wks.update([dataframe.columns.values.tolist()] + dataframe.values.tolist()) 



