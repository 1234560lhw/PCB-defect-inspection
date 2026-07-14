from ultralytics import YOLO
import torch

if __name__ == "__main__":
    print("GPU：", torch.cuda.get_device_name(0))
    model = YOLO("yolov3.yaml")
    res = model.train(
        data="./config/pcb.yaml",
        epochs=100,
        batch=8,
        device=0,
        imgsz=640,
        lr0=0.0005,
        pretrained=False,
        mosaic=1.0,
        plots=True,
        project="./runs/train",
        name="pcb_yolov3_exp2_lr0.0005"
    )
    print(f"低学习率实验mAP@0.5：{res.box.map50:.4f}")