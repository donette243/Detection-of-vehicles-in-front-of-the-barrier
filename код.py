import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  

import cv2, numpy as np, pygame, winsound
from PIL import ImageFont, ImageDraw, Image

video_path   = r"C:/Users/Donete/Documents/AirSim/TESTDK/cvtest.avi"
weights_path = "yolov3.weights"
config_path  = "yolov3.cfg"
names_path   = "coco.names"
alert_path   = "alert.wav"  # По желанию: индивидуальный звук

for file in [video_path, weights_path, config_path, names_path]:
    if not os.path.exists(file):
        raise FileNotFoundError(f" Файл не найден : {file}")

# Инициализация звука 
pygame.mixer.init()
use_wav = os.path.exists(alert_path)
if use_wav:
    try:
        pygame.mixer.music.load(alert_path)
    except pygame.error:
        print(" Файл alert.wav не воспроизводится: используется системный звуковой сигнал")
        use_wav = False

def play_alert():
    if use_wav:
        pygame.mixer.music.play()
    else:
        winsound.Beep(1000, 700)  # Частота (Гц), длительность (мс)

# Параметры области интереса (ROI: Region of Interest)
roi_rel = (0.0, 0.35, 1.0, 1.0)

# Загрузка модели YOLO
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open(names_path) as f:
    classes = [c.strip() for c in f]
VEHICLE_CLASSES = {'car', 'motorcycle'}

layers = net.getLayerNames()
out_layers = [layers[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Видео
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(" Невозможно открыть видео ")


fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter('result.avi', fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))
last_vehicle_count = 0  # Количество транспортных средств на предыдущем кадре
# цикл
while True:
    ok, frame = cap.read()
    if not ok:
        print(" Видео закончено")
        break

    h, w = frame.shape[:2]
    x1, y1 = int(w * roi_rel[0]), int(h * roi_rel[1])
    x2, y2 = int(w * roi_rel[2]), int(h * roi_rel[3])
    roi = frame[y1:y2, x1:x2]

    blob = cv2.dnn.blobFromImage(roi, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(out_layers)

    vehicles = []

    for output in outputs:
        for det in output:
            confs = det[5:]
            cid = int(np.argmax(confs))
            conf = confs[cid]
            if conf > 0.2 and classes[cid] in VEHICLE_CLASSES:
                cx = int(det[0] * (x2 - x1)) + x1
                cy = int(det[1] * (y2 - y1)) + y1
                bw = int(det[2] * (x2 - x1))
                bh = int(det[3] * (y2 - y1))
                bx = cx - bw // 2
                by = cy - bh // 2
                vehicles.append((bx, by, bw, bh, cid))

    # Оповещение при появлении нового транспортного средства
    cur_count = len(vehicles)
    if cur_count > last_vehicle_count:
        play_alert()
    last_vehicle_count = cur_count

    # Отображение обнаружений
    for bx, by, bw, bh, cid in vehicles:
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.putText(frame, classes[cid], (bx, by - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

   
    msg, col = (
        ("Транспортное средство перед шлагбаумом", (0, 255, 0))
        if cur_count else
        ("Нет транспортных средств перед шлагбаумом", (0, 0, 255))
    )

    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 32)
    draw.text((30, 40), msg, font=font, fill=col)
    frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    out.write(frame)

    # вывод 
    cv2.imshow("Vehicle Detection", frame)
    if cv2.waitKey(1) in (ord('q'), 27):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
