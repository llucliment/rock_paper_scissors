import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras import layers, models

# ── Rebuild model & load weights from existing .h5 ────────────────

model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.6),
    layers.Dense(3, activation='softmax')
])

model.load_weights('rps_model_da.h5', by_name=False, skip_mismatch=False)
print("Weights loaded!")

LABELS = ['Rock', 'Paper', 'Scissors']
COLORS = {'Rock': (60, 60, 220), 'Paper': (40, 180, 40), 'Scissors': (0, 180, 220)}

# ── MediaPipe setup ────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# ── Webcam loop ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # ── Bounding box from landmarks ────────────────────────
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            PADDING = 30
            x1 = max(0, int(min(x_coords) * w) - PADDING)
            y1 = max(0, int(min(y_coords) * h) - PADDING)
            x2 = min(w, int(max(x_coords) * w) + PADDING)
            y2 = min(h, int(max(y_coords) * h) + PADDING)

            # ── Crop & predict ─────────────────────────────────────
            roi = rgb[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_resized  = cv2.resize(roi, (64, 64))
            roi_norm     = roi_resized.astype('float32') / 255.0
            input_tensor = np.expand_dims(roi_norm, axis=0)

            preds      = model.predict(input_tensor, verbose=0)[0]
            class_idx  = np.argmax(preds)
            label      = LABELS[class_idx]
            confidence = preds[class_idx] * 100
            color      = COLORS[label]

            # ── Bounding box ───────────────────────────────────────
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ── Label pill ─────────────────────────────────────────
            text      = f'{label}  {confidence:.0f}%'
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cv2.rectangle(frame,
                          (x1, y1 - text_size[1] - 16),
                          (x1 + text_size[0] + 8, y1),
                          color, -1)
            cv2.putText(frame, text,
                        (x1 + 4, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # ── Hand landmarks ─────────────────────────────────────
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=color, thickness=2)
            )

    else:
        cv2.putText(frame, 'No hand detected',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (100, 100, 100), 2)

    cv2.imshow('Rock Paper Scissors', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()