from ultralytics import YOLO

student = YOLO(model = "/home/mas-server/distil/ultralytics/ultralytics/cfg/models/v8/yolov8n.yaml", task="detect_distill")
teacher_model = YOLO("yolov8m.pt", task="detect_distill")
student.train(data="coco.yaml", epochs=10, cache=True, kwargs={'teacher':teacher_model, 'embeds_s' : None, 'embeds_t' : None})
#eblan = YOLO(model = "/home/mas-server/distil/ultralytics/ultralytics/cfg/models/v8/yolov8n.yaml", task="detect")
#eblan.train(data="coco.yaml", epochs=10)