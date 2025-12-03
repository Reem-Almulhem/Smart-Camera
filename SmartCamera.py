# import os
# import cv2
# import numpy as np
# from PIL import Image
# from scipy.spatial.distance import cosine
# import torch
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from datetime import timedelta
# import csv

# # ========== إعدادات ==========
# CAMERA_INDEX = "rtsp://admin:7services@192.168.8.69:554/Streaming/Channels/101/"
# OUTPUT_VIDEO = r"C:\Users\LENOVO\Desktop\camera\output\output_labeled.mp4"
# CSV_LOG = r"C:\Users\LENOVO\Desktop\camera\output\event_log.csv"
# DATASET_DIR = r"C:\Users\LENOVO\Desktop\camera\employees"
# SIGNATURE = "Implemented by: Reem Almulhem"
# SIMILARITY_THRESHOLD = 0.60  # فقط من تتجاوز هذه النسبة يعتبر معروف

# # ========== الجهاز ==========
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # ========== تحميل النماذج ==========
# mtcnn = MTCNN(keep_all=True, image_size=160, margin=20, device=device)
# resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# # ========== تحميل الوجوه ==========
# known_faces = {}  # {person: [emb1, emb2, ...]}

# for person_name in os.listdir(DATASET_DIR):
#     person_dir = os.path.join(DATASET_DIR, person_name)
#     embeddings = []

#     if os.path.isdir(person_dir):
#         for file_name in os.listdir(person_dir):
#             file_path = os.path.join(person_dir, file_name)
#             try:
#                 img = Image.open(file_path).convert("RGB")
#                 face = mtcnn(img)
#                 if face is not None:
#                     if face.ndim == 3:
#                         face = face.unsqueeze(0)
#                     elif face.ndim > 4:
#                         continue
#                     emb = resnet(face.to(device)).detach().cpu().numpy().flatten()
#                     embeddings.append(emb)
#             except Exception as e:
#                 print(f"❌ Error loading {file_name}: {e}")

#     if embeddings:
#         known_faces[person_name] = embeddings
#         print(f"✅ Loaded {len(embeddings)} embeddings for {person_name}")
#     else:
#         print(f"⚠️ No valid faces found for {person_name}")

# # ========== تتبع الأحداث ==========
# active_people = {}  # {person: frame_index}
# event_log = []      # [time, event, person, duration]

# def frame_to_time(frame_idx, fps):
#     return str(timedelta(seconds=frame_idx / fps))

# # ========== فتح الكاميرا ==========
# cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_FFMPEG)
# fps = cap.get(cv2.CAP_PROP_FPS) or 10
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # تأكد أن مجلد الإخراج موجود
# os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
# out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# frame_idx = 0
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Failed to read frame")
#         continue

#     frame_idx += 1
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     pil_img = Image.fromarray(rgb)
#     boxes, _ = mtcnn.detect(pil_img)
#     faces = mtcnn(pil_img)

#     current_detected = set()

#     if boxes is not None and faces is not None:
#         for i, face in enumerate(faces):
#             if face is None:
#                 continue

#             if face.ndim == 3:
#                 face = face.unsqueeze(0)
#             elif face.ndim > 4:
#                 continue

#             emb = resnet(face.to(device)).detach().cpu().numpy().flatten()

#             best_similarity = 0.0
#             matched_person = None
#             for person, embs in known_faces.items():
#                 for known_emb in embs:
#                     dist = cosine(emb, known_emb)
#                     similarity = 1 - dist
#                     if similarity > best_similarity:
#                         best_similarity = similarity
#                         matched_person = person

#             if matched_person and best_similarity >= SIMILARITY_THRESHOLD:
#                 person_label = matched_person
#                 label = f"Known: {matched_person.capitalize()} ({best_similarity:.2%})"
#                 color = (0, 255, 0)
#             else:
#                 person_label = "Unknown"
#                 label = "Unknown"
#                 color = (0, 0, 255)

#             current_detected.add(person_label)
#             x1, y1, x2, y2 = map(int, boxes[i])
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

#     # سجل الظهور والاختفاء
#     for person in list(active_people.keys()):
#         if person not in current_detected:
#             start_frame = active_people.pop(person)
#             duration = (frame_idx - start_frame) / fps
#             event_log.append([frame_to_time(frame_idx, fps), "Disappeared", person, f"{duration:.2f} sec"])

#     for person in current_detected:
#         if person not in active_people:
#             active_people[person] = frame_idx
#             event_log.append([frame_to_time(frame_idx, fps), "Appeared", person, ""])

#     # التوقيع
#     cv2.putText(frame, SIGNATURE, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     out.write(frame)
#     cv2.imshow("Smart Camera", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# out.release()
# cv2.destroyAllWindows()

# # ========== حفظ السجل ==========
# os.makedirs(os.path.dirname(CSV_LOG), exist_ok=True)
# with open(CSV_LOG, mode="w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Time", "Event", "Person", "Duration"])
#     writer.writerows(event_log)

# print("📂 تم حفظ سجل الأحداث في:", CSV_LOG)
# print("🎬 تم حفظ الفيديو النهائي:", OUTPUT_VIDEO)
import cv2
import os
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
import threading
import time

# ============================
# إعدادات
# ============================
RTSP_URL = "rtsp://admin:7services@192.168.8.69:554/Streaming/Channels/101/"
DATASET_DIR = r"C:\Users\LENOVO\Desktop\camera\employees"

SIMILARITY_THRESHOLD = 0.60
DETECT_EVERY_N_FRAMES = 10     # التعرف كل 10 فريمات فقط (أسرع بكثير)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================
# تحميل النماذج
# ============================
mtcnn = MTCNN(keep_all=True, image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ============================
# تحميل الوجوه (علي – سارة)
# ============================
known_faces = {}

for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)
    embeddings = []

    if os.path.isdir(person_dir):
        for file_name in os.listdir(person_dir):
            file_path = os.path.join(person_dir, file_name)
            try:
                img = Image.open(file_path).convert("RGB")
                face = mtcnn(img)
                if face is not None:
                    if face.ndim == 3:
                        face = face.unsqueeze(0)

                    emb = resnet(face.to(device)).detach().cpu().numpy().flatten()
                    embeddings.append(emb)

            except Exception as e:
                print(f"Error loading {file_name}: {e}")

    if embeddings:
        known_faces[person_name] = embeddings
        print(f"Loaded {len(embeddings)} embeddings for {person_name}")

# ============================
# قراءة الكاميرا بخيط منفصل (Thread)
# ============================
class CameraThread:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.ret = False
        self.frame = None
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()
            time.sleep(0.005)

    def stop(self):
        self.running = False
        self.cap.release()


camera = CameraThread(RTSP_URL)
time.sleep(1)

frame_count = 0

# ============================
# الحلقة الرئيسية
# ============================
while True:
    if not camera.ret:
        continue

    frame = camera.frame.copy()
    frame_count += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # التعرف فقط كل N فريمات
    if frame_count % DETECT_EVERY_N_FRAMES == 0:
        boxes, _ = mtcnn.detect(pil_img)
        faces = mtcnn(pil_img)
    else:
        boxes = None
        faces = None

    if boxes is not None and faces is not None:
        for i, face in enumerate(faces):
            if face is None:
                continue

            if face.ndim == 3:
                face = face.unsqueeze(0)

            emb = resnet(face.to(device)).detach().cpu().numpy().flatten()

            best_similarity = 0
            matched_person = "Unknown"

            for person, embs in known_faces.items():
                for known_emb in embs:
                    similarity = 1 - cosine(emb, known_emb)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        matched_person = person

            if best_similarity >= SIMILARITY_THRESHOLD:
                label = f"{matched_person} ({best_similarity:.2f})"
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Smart Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.stop()
cv2.destroyAllWindows()
