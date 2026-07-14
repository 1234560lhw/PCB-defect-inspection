from ultralytics import YOLO

# 训练好的best.pt
model = YOLO(r"D:\project\project-PCB board defect inspection\runs\detect\runs\train\pcb_yolov3_exp1_round_2-2\weights\best.pt")

# 批量推理验证集图片，自动保存带缺陷框效果图
results = model.predict(
    source="./dataset/images/val",
    save=True,
    save_txt=True,
    save_conf=True,
    conf=0.25
)
print("推理完成，检测图片保存在 runs/detect/predict 文件夹")

# 打印单张检测指标
for r in results[:3]:
    print("检测缺陷框坐标：", r.boxes.xyxy.cpu().numpy())
    print("缺陷类别ID：", r.boxes.cls.cpu().numpy())
    print("置信度：", r.boxes.conf.cpu().numpy())