import numpy as np
import math
import cv2


def rotate_image(image, rotate_angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, rotate_angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def image_processing_and_contours(image):
    clahe = cv2.createCLAHE(clipLimit=50, tileGridSize=(20, 20))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((2, 2), np.uint8)
    obr_img = cv2.erode(thresh, kernel, iterations=2)

    obr_img = cv2.GaussianBlur(obr_img, (3, 3), 0)

    contours, hierarchy = cv2.findContours(obr_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    return contours


def image_to_cells(image_name):
    img = cv2.imread(image_name)

    all_contours = image_processing_and_contours(img)

    max_area = 0
    max_angle = 0
    # перебираем все найденные контуры в цикле для нахождения самого большого прямоугольника
    for cnt in all_contours:
        rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат
        area = int(rect[1][0] * rect[1][1])  # вычисление площади
        center = (int(rect[0][0]), int(rect[0][1]))

        # вычисление координат двух векторов, являющихся сторонам прямоугольника
        edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
        edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))

        # выясняем какой вектор больше
        usedEdge = edge1
        if cv2.norm(edge2) > cv2.norm(edge1):
            usedEdge = edge2
        reference = (1, 0)  # горизонтальный вектор, задающий горизонт

        # вычисляем угол между самой длинной стороной прямоугольника и горизонтом
        angle = 180.0 / math.pi * math.acos(
            (reference[0] * usedEdge[0] + reference[1] * usedEdge[1]) / (cv2.norm(reference) * cv2.norm(usedEdge)))

        if area > max_area and (0 <= angle <= 5 or 85 <= angle <= 90):
            max_area = area
            if angle >= 80:
                max_angle = abs(angle - 90)
            else:
                max_angle = angle

    img2 = rotate_image(img, -max_angle).copy()
    all_contours = image_processing_and_contours(img2)

    max_area = 0
    max_box = []
    max_angle = 0
    max_rect = 0
    # перебираем все найденные контуры в цикле для нахождения самого большого прямоугольника
    for cnt in all_contours:
        rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат
        area = int(rect[1][0] * rect[1][1])  # вычисление площади
        center = (int(rect[0][0]), int(rect[0][1]))
        # вычисление координат двух векторов, являющихся сторонам прямоугольника
        edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
        edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))

        # выясняем какой вектор больше
        usedEdge = edge1
        if cv2.norm(edge2) > cv2.norm(edge1):
            usedEdge = edge2
        reference = (1, 0)  # горизонтальный вектор, задающий горизонт

        # вычисляем угол между самой длинной стороной прямоугольника и горизонтом
        angle = 180.0 / math.pi * math.acos(
            (reference[0] * usedEdge[0] + reference[1] * usedEdge[1]) / (cv2.norm(reference) * cv2.norm(usedEdge)))

        if area > max_area and (0 <= angle <= 5 or 85 <= angle <= 90):
            max_area = area
            max_box = box
            max_rect = rect
            if angle >= 80:
                max_angle = abs(angle - 90)
            else:
                max_angle = angle

    img3 = img2[min(max_box[:, 1]):max(max_box[:, 1]), min(max_box[:, 0]):max(max_box[:, 0])].copy()
    tmp_img3 = img3.copy()
    all_contours = image_processing_and_contours(img3)

    all_boxes = []
    # перебираем все найденные контуры в цикле
    for cnt in all_contours:
        rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)# округление координат
        area = int(rect[1][0] * rect[1][1])  # вычисление площади
        center = (int(rect[0][0]), int(rect[0][1]))

        # вычисление координат двух векторов, являющихся сторонам прямоугольника
        edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
        edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))

        # выясняем какой вектор больше
        usedEdge = edge1
        if cv2.norm(edge2) > cv2.norm(edge1):
            usedEdge = edge2
        reference = (1, 0)  # горизонтальный вектор, задающий горизонт

        # вычисляем угол между самой длинной стороной прямоугольника и горизонтом
        angle = 180.0 / math.pi * math.acos(
            (reference[0] * usedEdge[0] + reference[1] * usedEdge[1]) / (cv2.norm(reference) * cv2.norm(usedEdge)))

        if (abs(angle - max_angle) < 2 or abs(abs(angle - 90) - max_angle) < 2) and\
                (int(rect[1][1]) >= int(max_rect[1][1] / 80) and int(rect[1][0]) >= int(max_rect[1][0] / 80)):
            cv2.drawContours(img3, [box], 0, (255, 0, 0), 2)
            if area < 0.98 * max_area:
                tmp_box = box
                tmp_box[0] = [min(box[:, 0]), min(box[:, 1])]
                tmp_box[1] = [min(box[:, 0]), max(box[:, 1])]
                tmp_box[2] = [max(box[:, 0]), min(box[:, 1])]
                tmp_box[3] = [max(box[:, 0]), max(box[:, 1])]
                all_boxes.append(tmp_box)

    min_x = 10000
    max_x = 0
    for box in all_boxes:
        if box[0][0] < min_x:
            min_x = box[0][0]
        if box[2][0] > max_x:
            max_x = box[2][0]
    small_diffx = (max_x - min_x) / 200
    left_boxes = []
    left_y = []
    last_y = 0
    for box in all_boxes:
        if min_x <= box[0][0] <= min_x + small_diffx:
            left_boxes.append(box)
            left_y.append(box[0][1])
            if last_y < box[1][1]:
                last_y = box[1][1]
    left_y.append(last_y)
    left_y.sort()
    amount_of_rows = len(left_boxes)

    '''
    for box in left_boxes:
        cv2.imshow("left box", tmp_img3[min(box[:, 1]):max(box[:, 1]), min(box[:, 0]):max(box[:, 0])])
        cv2.waitKey()
    '''

    min_y = 10000
    max_y = 0
    for box in all_boxes:
        if box[0][1] < min_y:
            min_y = box[0][1]
        if box[3][1] > max_y:
            max_y = box[3][1]
    small_diffy = (max_y - min_y) / 200
    upper_boxes = []
    upper_x = []
    last_x = 0
    for box in all_boxes:
        if min_y <= box[0][1] <= min_y + small_diffy:
            upper_boxes.append(box)
            upper_x.append(box[0][0])
            if last_x < box[2][0]:
                last_x = box[2][0]
    upper_x.append(last_x)
    upper_x.sort()
    amount_of_columns = len(upper_boxes)

    cells = []
    for i in range(amount_of_rows):
        cells.append([])
        for j in range(amount_of_columns):
            cells[i].append(tmp_img3[left_y[i]:left_y[i + 1], upper_x[j]:upper_x[j + 1]])

    for i in range(amount_of_rows):
        for j in range(amount_of_columns):
            height, width = cells[i][j].shape[:2]
            cells[i][j] = cells[i][j][int(height * 0.02):int(height * 0.98), int(width * 0.02):int(width * 0.98)]

    '''
    for i in range(len(cells)):
        for j in range(len(cells[i])):
            cv2.imshow('cells', cells[i][j])
            cv2.waitKey()
    '''
    return cells