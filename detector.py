import cv2
import numpy as np
from math import sqrt


color_dict = {0: (255, 255, 255),  # white
                1: (50, 47, 47),  # black
                2: (85, 95, 230),  # red
                3: (75, 160, 245),  # orange
                4: (79, 240, 240),  # yellow
                5: (110, 190, 5),  # green
                6: (250, 220, 5),  # cyan
                7: (229, 135, 5),  # blue
                8: (228, 151, 95),  # violet
                9: (210, 162, 220)}  # magenta


# helper functions
def minn(val):
  return max(val - thresh, 0)


def maxn(val):
  return min(val + thresh, 255)


def check(d, val):
  for k, v in d.items():
    if v == val:
      return k
  return -1


# cut out the contoured part of image
def cutout(image, contour):
  pts = np.float32([contour[0][0], contour[3][0], contour[1][0], contour[2][0]][::-1])
  pts2 = np.float32([[0, 0], [200, 0], [0, 200], [200, 200]])
  M = cv2.getPerspectiveTransform(pts, pts2)
  dst = cv2.warpPerspective(image, M, (200, 200))
  return dst


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
 

# KNN implementation
def get_neighbors(train, test_row, num_neighbors):
  distances = list()
  for train_row in train:
    dist = euclidean_distance(test_row, train_row)
    distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
  neighbors = list()
  for i in range(num_neighbors):
    neighbors.append(distances[i][0])
  return neighbors


# color code generation
def color_encode(dst):
    code = ''
    for i in range(2):
        for j in range(2):
            bgr = dst[i * 100 + 50][j * 100 + 50]
            idk = get_neighbors(list(color_dict.values()), np.array(bgr), 1)[0]
            code += str(check(color_dict, idk))
    return code


# color picking
def encode(code):
    if (int(code[0]) + int(code[1]) == 11) and (int(code[0]) + int(code[2]) != 11):
        return code
    elif (int(code[1]) + int(code[3]) == 11) & (int(code[1]) + int(code[0]) != 11):
        return code[1] + code[3] + code[0] + code[2]
    elif (int(code[2]) + int(code[0]) == 11) & (int(code[2]) + int(code[3]) != 11):
        return code[2] + code[0] + code[3] + code[1]
    else:
        return code[::-1]
  


camera = cv2.VideoCapture(0)
for i in range(100):
  camera.read()
  cv2.waitKey(1)
_, img = camera.read()
cv2.waitKey(1)
i = img.copy()[100:630, 400:760]

k = 13  # kernel
i = cv2.GaussianBlur(i, (k, k), 101)
thresh = 42
arr = []

# get binary mask for each color
for v in color_dict.values():
  i2 = cv2.inRange(i, (minn(v[0]), minn(v[1]), minn(v[2])), (maxn(v[0]), maxn(v[1]), maxn(v[2]))) 
  arr.append(i2)

# combine masks
result = sum(arr)
result = result.clip(0, 255).astype("uint8")

# make masks rectangle-like
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
opening = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 7))
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 14))
closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

contours, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
a_cnts = []
tr_cnts = []
tr = 3803  # the code we're searching for
for j in range(len(contours)):
  epsilon = 0.07 * cv2.arcLength(contours[j], True)
  print('len', cv2.arcLength(contours[j], True))
  a_cnts.append(cv2.approxPolyDP(contours[j], epsilon, True))
  print(len(a_cnts[-1]))
  try:
    res_col = encode(color_encode(cutout(i, a_cnts[-1])))
    if res_col[0]==res_col[1]==res_col[2]==res_col[3] or cv2.arcLength(contours[j], True) <= 320:
      del a_cnts[-1]
      continue
    cv2_imshow(cutout(i, a_cnts[-1]))
    if res_col == '8349':
      res_col = '8347'
    if tr == res_col:
      tr_cnts.append(a_cnts[-1])
    print(res_col)
  except: del a_cnts[-1]
