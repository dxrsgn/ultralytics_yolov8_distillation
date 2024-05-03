from ultralytics import YOLO

student = YOLO(model = "/home/mas-server/distil/ultralytics/ultralytics/cfg/models/v8/distillyolov8n.yaml", task="detect_distill")
teacher_model = YOLO("yolov8m.pt", task="detect_distill")
student.train(data="coco.yaml", epochs=20, cache=True, name="reg1weight0.5heads3", kwargs={'teacher':teacher_model, 'embeds_s' : [15,18,21], 'embeds_t' : [15,18,21]})
#eblan = YOLO(model = "/home/mas-server/distil/ultralytics/ultralytics/cfg/models/v8/yolov8n.yaml", task="detect")
#eblan.train(data="coco.yaml", cache=True, epochs=20)