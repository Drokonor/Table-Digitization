import numpy as np
import os
import cv2
from Cell_splitting import image_to_cells
import keras
import pandas as pd
from datetime import datetime
import time

model = keras.models.load_model('comnist_letters.h5')

comnist_labels = []
for i in range(42):
    comnist_labels.append(i)
comnist_labels_char = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 1040, 1041, 1042, 1043, 1044, 1045,  1046, 1047, 1048,
                       1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064,
                       1065, 1066, 1067, 1068, 1069, 1070, 1071]


def emnist_predict_img(model, img):
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr / 255.0
    img_arr = img_arr.reshape((1, 28, 28, 1))
    predict = model.predict([img_arr])
    result = np.argmax(predict, axis=1)
    return chr(comnist_labels_char[comnist_labels[result[0]]])


dir = os.path.abspath(os.curdir)
files = os.listdir(dir + '\\Example_Tables')
cells_of_tables = []
count = 0
start_time = datetime.now()
for i in range(len(files)):
    cells_of_tables.append(image_to_cells(dir + '\\Example_Tables\\' + files[i]))
for k in range(len(cells_of_tables)):
    table_df = []
    cells = cells_of_tables[k]
    for i in range(len(cells)):
        table_df.append([])
        for j in range(len(cells[i])):
            imgOriginal = cells[i][j]
            img_cp = np.array(imgOriginal)
            gray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
            blur = cv2.medianBlur(gray, 5)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 8)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
            dilate = cv2.dilate(thresh, kernel, iterations=10)

            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            output = imgOriginal.copy()
            height, width = thresh.shape[:2]
            letters = []
            for idx, contour in enumerate(contours):
                (x, y, w, h) = cv2.boundingRect(contour)
                if 0.03 * width < w < 0.9 * width and h > 0.1 * height:
                    cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
                    letter_crop = thresh[y:y + h, x:x + w]
                    size_max = max(w, h)
                    letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
                    if w > h:
                        y_pos = size_max // 2 - h // 2
                        letter_square[y_pos:y_pos + h, 0:w] = letter_crop
                    elif w < h:
                        x_pos = size_max // 2 - w // 2
                        letter_square[0:h, x_pos:x_pos + w] = letter_crop
                    else:
                        letter_square = letter_crop

                    letters.append((x, w, cv2.resize(letter_square, (28, 28), interpolation=cv2.INTER_AREA)))

            letters.sort(key=lambda x: x[0], reverse=False)
            '''
            for k in range(len(letters)):
                cv2.imshow("letters", letters[k][2])
                cv2.waitKey(0)
            
            cv2.imshow("Input", cells[i][j])
            cv2.imshow("Enlarged", thresh)
            cv2.imshow("Output", output)
            cv2.waitKey(0)
            '''
            result = ''
            for x in range(len(letters)):
                dn = letters[x + 1][0] - letters[x][0] - letters[x][1] if x < len(letters) - 1 else 0
                result += emnist_predict_img(model, letters[x][2])
            table_df[i].append(result)
    df = pd.DataFrame(table_df)
    df.to_excel(dir + '\\Comnist_Example_Tables\\' + files[k][:-3] + 'xls')
    count += 1
    print(count)
    print(datetime.now() - start_time)
