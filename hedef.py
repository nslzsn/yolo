from ultralytics import YOLO
import cv2

# Model yükle
model = YOLO("best.pt")  # Kendi eğittiğin model burada olacak

# Webcam başlat (0: default kamera)
cap = cv2.VideoCapture(0)

# Hedef sınıf
TARGET_CLASS = "person"
COCO_CLASSES = model.names  # Modelden sınıf isimlerini alıyoruz

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntü merkezini hesapla
    h, w, _ = frame.shape
    center_x, center_y = w // 2, h // 2

    # Model tahmini
    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = COCO_CLASSES[cls_id]

            if class_name == TARGET_CLASS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_center_x = (x1 + x2) // 2
                bbox_center_y = (y1 + y2) // 2

                # Kırmızı dikdörtgen ve merkez noktası
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(frame, (bbox_center_x, bbox_center_y), 5, (0, 0, 255), -1)

                # Offset hesaplama
                offset_x = bbox_center_x - center_x
                offset_y = bbox_center_y - center_y

                # Sağ üst köşeye yazdırma kutusu
                info_text = f"Offset X: {offset_x}  Y: {offset_y}"
                (text_w, text_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (10, 10), (10 + text_w + 10, 10 + text_h + 20), (50, 50, 50), -1)
                cv2.putText(frame, info_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Anlık görüntü göster
    cv2.imshow("Kamera Takip", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' çıkış
        break

cap.release()
cv2.destroyAllWindows()
