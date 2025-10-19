import numpy as np
import cv2
cap = cv2.VideoCapture("video.mp4")
# 顯示視訊相關資訊
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Resolution: {width} x {height}") # Frame 解析度
print(f"FPS: {fps:.2f}") # frame-per-second (fps)
# 播放影片
while True:
  ret, frame = cap.read()
  if not ret:
    break
  cv2.imshow("Frame", frame)
  if cv2.waitKey(int(1000 // fps)) & 0xff == 27: # 使用 ESC 提早結束
    break
# 關閉視訊檔案與釋放資源
cap.release()
cv2.destroyAllWindows()