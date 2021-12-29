from mafaextractor import extract_mafa
import pandas as pd

'''
Annotation MAFA:
img_name, x_face_min, y_face_min, face_width, face_height, left_eye_x, left_eye_y, right_eye_x, right_eye_y, occ_width, 
occ_height, occ_type, occ_degree, gender, race, orientation, glasses_width, glasses_height, x_face_max, y_face_max, 
x_occ_min, y_occ_min, x_occ_max, y_occ_max, x_glasses_min, x_glasses_max, y_glasses_min, y_glasses_max, 

Annotation RetinaFace:
# path
x_min, y_min, width, height, left_eye_x, left_eye_y, visibleness, right_eye_x, right_eye_y, visible, 
nose_y, nose_x, visible, left_mouth_x, left_mouth_y, visible, right_mouth_x, right_mouth_y, visible, blur

array(['train_00000001.jpg'], dtype='<U18'), array([[ 95, 160,  91,  91, 113, 177, 158, 172,   7,  26,  82,  89,   1,
                  3,   1,   1,   3,  -1,  -1,  -1,  -1]], dtype=int16))                                                                                          ,
'''

df = extract_mafa("/home/tony/Documents/CAP/dataset/MAFA/MAFA-Label-Train/LabelTrainAll.mat")

# df_p = df.head(100)

bbox_elements = ['x_face_min', 'y_face_min', 'face_width', 'face_height']
left_eye_elements = ['left_eye_x', 'left_eye_y']
right_eye_elements = ['right_eye_x', 'right_eye_y']
default_zeroes = str("-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 0.5\n")
path = str()
previous_path = str()
result = str()
f = open("label.txt", "w")

for index, row in df.iterrows():
    path = str("# mafa/" + row['img_name'] + "\n")
    if path != previous_path:
        result += path
    for e in bbox_elements:
        result += str(str(row[e]) + " ")
    for e in left_eye_elements:
        result += str(str(row[e] * 1.0) + " ")
    result += str("0.0 ")
    for e in right_eye_elements:
        result += str(str(row[e] * 1.0) + " ")
    result += str("0.0 ")
    result += default_zeroes
    f.write(result)
    result = str()
    previous_path = path


# path = df_p.head(1).get('img_name')[0]
#
#
#
# f.write(str("# " + path))
