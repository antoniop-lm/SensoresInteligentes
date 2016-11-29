# Antonio Pedro Lavezzo Mazzarolo - 8626232
import numpy as np
import cv2
import pygame

# Comentario meu for dummies (como rodar na minha rasp):
# source ~/.profile
# workon opencv
# toggleglobalsitepackages

def calcpts(oldpts,pts,index):
  total = oldpts[0][index] - pts[0][index]
  total += oldpts[1][index] - pts[1][index]
  total += oldpts[2][index] - pts[2][index]
  total += oldpts[3][index] - pts[3][index]
  return total/4

cap = cv2.VideoCapture(0)
ret,frame = cap.read()

# setup initial location of window
# a ROI inicial foi criada usando toda a area do frame capturado para evitar bugs de tracking (640 x 480)
r,h,c,w = 0,480,0,640
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# varias opcoes de mask por conta da luminosidade (H - 180, S - 255, V - 255)
#mask = cv2.inRange(hsv_roi, np.array((8.5, 112.2,99.45)), np.array((180.,255.,255.)))
#mask = cv2.inRange(hsv_roi, np.array((4, 122.4,124.95)), np.array((180.,255.,255.)))
#mask = cv2.inRange(hsv_roi, np.array((170., 25.5,117.3)), np.array((180.,255.,255.)))
mask = cv2.inRange(hsv_roi, np.array((0., 60.,192.)), np.array((180.,255.,255.)))
#mask = cv2.inRange(hsv_roi, np.array((10.5, 102.,181.05)), np.array((180.,255.,255.)))

roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
# numero de iteracoes definidas para o algoritmo de camshift
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

oldpts = [[0,0],[0,0],[0,0],[0,0]]
vol = 0.5
pygame.init()
pygame.mixer.music.load('Gorrillaz-FeelGoodInc.ogg')
pygame.mixer.music.play(-1, 0.0)
pygame.mixer.music.set_volume(vol)

while(1):
  ret ,frame = cap.read()

  if ret == True:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

    # apply camshift to get the new location
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    # Draw it on image
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    x = calcpts(oldpts,pts,0)
    y = calcpts(oldpts,pts,1) 
    if x > 50:
      print("direita")
      pygame.mixer.music.unpause()
    elif x < -50:
      print("esquerda")
      pygame.mixer.music.pause()
    if y > 50:
      print("up")
      vol = vol + 0.1
      if vol > 1.0:
        vol = 1.0
      pygame.mixer.music.set_volume(vol)
    elif y < -50:
      print("down")
      vol = vol - 0.1
      if vol < 0.0:
        vol = 0.0
      pygame.mixer.music.set_volume(vol)
    oldpts = pts
    img2 = cv2.polylines(frame,[pts],True, 255,2)
    cv2.imshow('img2',img2)

    k = cv2.waitKey(60) & 0xff
    if k == 27:
      break

  else:
    break

pygame.mixer.music.stop()
cv2.destroyAllWindows()
cap.release()