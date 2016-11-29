import numpy as np
import cv2

def calc(oldpts,pts,index):
  total = oldpts[0][index] - pts[0][index]
  total += oldpts[1][index] - pts[1][index]
  total += oldpts[2][index] - pts[2][index]
  total += oldpts[3][index] - pts[3][index]
  return total/4

cap = cv2.VideoCapture(0)
count = 0

while(1):
  ret ,frame = cap.read()

  if ret == True:
    cv2.imshow('img2',frame)

    k = cv2.waitKey(60) & 0xff
    if k == 27:
      break
    elif k == 113:
      cv2.imwrite(str(count)+".jpg",frame)
      count = count + 1

  else:
    break

cv2.destroyAllWindows()
cap.release()