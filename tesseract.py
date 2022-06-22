import numpy as np
import pandas as pd
import math
import os
import cv2
import pytesseract
import Example_Tables
from Cell_splitting import image_to_cells
from datetime import datetime
import time

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

dir = os.path.abspath(os.curdir)
all_table_names = []
files = os.listdir(dir + '\\Tables')
cells_of_tables = []
for i in range(len(files)):
    if 'jpg' or 'png' in files[i]:
        cells_of_tables.append(image_to_cells(dir + '\\Tables\\' + files[i]))

count = 0
start_time = datetime.now()
for i in range(len(cells_of_tables)):
    table_df = []
    for j in range(len(cells_of_tables[i])):
        table_df.append([])
        for k in range(len(cells_of_tables[i][j])):
            try:
                img = cells_of_tables[i][j][k].copy()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                config = r'--oem 3 --psm 6 -c tessedit_char_blacklist=_|[]'
                table_df[j].append(pytesseract.image_to_string(img, config=config, lang='rus'))
            except:
                table_df[j].append('')
    df = pd.DataFrame(table_df)
    df.to_excel(dir + '\\Tesseract_Tables\\' + files[i][:-3] + 'xls')
    count += 1
    print(count)
    print(datetime.now() - start_time)
