import cv2
import numpy as np
import csv
import tkinter as tk
from tkinter import filedialog




def getvideo():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])


def get_video_length():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps else 0
    cap.release()
    return duration

def get_fps():
    output_csv_path = "output.csv"
    # Apri il video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Errore nell'apertura del video.")
        return

    # Ottieni FPS e calcola quanti frame saltare per arrivare a 1 secondo
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Impossibile determinare il frame rate del video.")
        return

    return fps


def tracker():
    output_csv_path = "output.csv"
    # Apri il video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Errore nell'apertura del video.")
        return

    fps = get_fps()

    frame_count_to_skip = int(fps)
    for _ in range(frame_count_to_skip):
        success, frame = video.read()
        if not success:
            print("Errore nel saltare i frame iniziali.")
            return

    # Inizializza il MultiTracker
    multi_tracker = cv2.legacy.MultiTracker_create()

    # Selezione delle ROI da tracciare
    while True:
        bbox = cv2.selectROI("Seleziona oggetto da tracciare (ESC per finire)", frame, fromCenter=False)
        if bbox == (0, 0, 0, 0):
            break
        tracker = cv2.legacy.TrackerCSRT_create()
        multi_tracker.add(tracker, frame, bbox)

    cv2.destroyWindow("Seleziona oggetto da tracciare (ESC per finire)")

    # Apri il file CSV per scrivere le posizioni
    with open(output_csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame', 'oggetto_id', 'x', 'y', 'w', 'h'])

        frame_index = frame_count_to_skip  # inizia a contare dopo il salto

        # Loop di tracking
        while True:
            ret, frame = video.read()
            if not ret:
                break

            success, boxes = multi_tracker.update(frame)

            for i, newbox in enumerate(boxes):
                x, y, w, h = [int(v) for v in newbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Oggetto {i + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Scrivi nel CSV
                writer.writerow([frame_index, i + 1, x, y, w, h])

            cv2.imshow("Multi-Tracking", frame)
            frame_index += 1

            if cv2.waitKey(30) & 0xFF == 27:  # ESC per uscire
                break

    video.release()
    cv2.destroyAllWindows()
    print(f"Tracking completato. Risultati salvati in {output_csv_path}")


# def tracker():
#     output_csv_path = "output.csv"
#     # Apri il video
#     video = cv2.VideoCapture(video_path)
#     if not video.isOpened():
#         print("Errore nell'apertura del video.")
#         return
#
#     # Ottieni FPS e calcola quanti frame saltare per arrivare a 1 secondo
#     fps = video.get(cv2.CAP_PROP_FPS)
#     if fps <= 0:
#         print("Impossibile determinare il frame rate del video.")
#         return
#
#     frame_count_to_skip = int(fps)
#     for _ in range(frame_count_to_skip):
#         success, frame = video.read()
#         if not success:
#             print("Errore nel saltare i frame iniziali.")
#             return
#
#     # Inizializza il MultiTracker
#     multi_tracker = cv2.legacy.MultiTracker_create()
#
#     # Selezione delle ROI da tracciare
#     while True:
#         bbox = cv2.selectROI("Seleziona oggetto da tracciare (ESC per finire)", frame, fromCenter=False)
#         if bbox == (0, 0, 0, 0):
#             break
#         tracker = cv2.legacy.TrackerCSRT_create()
#         multi_tracker.add(tracker, frame, bbox)
#
#     cv2.destroyWindow("Seleziona oggetto da tracciare (ESC per finire)")
#
#     # Apri il file CSV per scrivere le posizioni
#     with open(output_csv_path, mode='w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['frame', 'oggetto_id', 'x', 'y', 'w', 'h'])
#
#         frame_index = frame_count_to_skip  # inizia a contare dopo il salto
#
#         # Loop di tracking
#         while True:
#             ret, frame = video.read()
#             if not ret:
#                 break
#
#             success, boxes = multi_tracker.update(frame)
#
#             for i, newbox in enumerate(boxes):
#                 x, y, w, h = [int(v) for v in newbox]
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 cv2.putText(frame, f"Oggetto {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
#                 # Scrivi nel CSV
#                 writer.writerow([frame_index, i + 1, x, y, w, h])
#
#             cv2.imshow("Multi-Tracking", frame)
#             frame_index += 1
#
#             if cv2.waitKey(30) & 0xFF == 27:  # ESC per uscire
#                 break
#
#     video.release()
#     cv2.destroyAllWindows()
#     print(f"Tracking completato. Risultati salvati in {output_csv_path}")

getvideo()

with open("fps.txt", 'w') as file:
    file.write(str(get_fps()))
print(f"Video lenght saved to fps.txt")

with open("video_lenght.txt", 'w') as file:
    file.write(str(get_video_length()))
print(f"Video lenght saved to video_lenght.txt")

tracker()
