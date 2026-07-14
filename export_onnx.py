from ultralytics import YOLO

if __name__ == "__main__":
    # 训练文件夹
    weight_path = r"D:\project\project-PCB board defect inspection\runs\detect\runs\train\pcb_yolov3_exp1_round_2-2\weights\best.pt"
    model = YOLO(weight_path)

    # 导出ONNX格式
    output_file = model.export(
        format="onnx",
        opset=12,  
        simplify=True,  
        imgsz=640
    )

    print(f"ONNX模型导出完成！保存路径：{output_file}")