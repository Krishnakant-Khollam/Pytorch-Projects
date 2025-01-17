from ultralytics import YOLO

model = YOLO("runs/segment/train/weights/best.pt")
model.predict(
    source=0,
    show=True,
    save=True,
    conf=0.7,
    line_width=2,
    show_labels=True,
    show_conf=True,
    classes=[0, 1],
)
