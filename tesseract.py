import numpy as np
import pandas as pd
import math
import os
import cv2
import pytesseract
import Example_Tables
from Cell_splitting import image_to_cells


pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

dir = os.path.abspath(os.curdir)
all_table_names = []
files = os.listdir(dir + '\\Example_Tables')
cells_of_tables = []
for i in range(len(files)):
    cells_of_tables.append(image_to_cells(dir + '\\Example_Tables\\' + files[i]))


for i in range(len(cells_of_tables)):
    table_df = []
    for j in range(len(cells_of_tables[i])):
        table_df.append([])
        for k in range(len(cells_of_tables[i][j])):
            img = cells_of_tables[i][j][k].copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            config = r'--oem 3 --psm 6'
            table_df[j].append(pytesseract.image_to_string(img, config=config, lang='rus'))
    df = pd.DataFrame(table_df)
    df.to_excel(dir + '\\Tesseract_Tables\\' + files[i][:-3] + 'xls')
