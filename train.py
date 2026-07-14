from ultralytics import YOLO
import torch
import os

if __name__ == "__main__":
    print("CUDA是否可用：", torch.cuda.is_available())
    print("GPU型号：", torch.cuda.get_device_name(0))
    print("CUDA版本：", torch.version.cuda)

    model = YOLO("yolov3.yaml", task="detect")

    train_result = model.train(
        data="./config/pcb.yaml",
        epochs=50,
        batch=4,
        device=0,
        imgsz=640,
        lr0=0.001,
        lrf=0.01,
        pretrained=False,
        mosaic=1.0,
        mixup=0,
        save=True,
        save_period=5,
        plots=True,
        project="./runs/train",
        name="pcb_yolov3_exp1_round_2",
        patience=10,
        workers=0
    )
    print("===== 训练完成 =====")
    print(f"最优权重路径：./runs/train/pcb_yolov3_exp1_round_2/weights/best.pt")
    print(f"最终mAP@0.5：{train_result.box.map50:.4f}")