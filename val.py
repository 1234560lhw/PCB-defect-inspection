from ultralytics import YOLO

if __name__ == "__main__":
    # 从你训练日志复制的真实best.pt绝对路径
    model_path = r"D:\project\project-PCB board defect inspection\runs\detect\runs\train\pcb_yolov3_exp1_round_2\weights\best.pt"
    model = YOLO(model_path)

    # 评估数据集，batch和训练保持4，指定输出文件夹
    metrics = model.val(
        data="./config/pcb.yaml",
        batch=4,
        project="runs/detect",   # 顶层目录固定runs/detect
        name="luhengwu",         # 子文件夹名称luhengwu
        exist_ok=True            # 文件夹存在直接覆盖，不生成新后缀文件夹
    )

    print("==========模型完整评估结果==========")
    print(f"全局mAP@0.5: {metrics.box.map50:.4f}")
    print(f"全局mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"平均精确率P: {metrics.box.p.mean():.4f}")
    print(f"平均召回率R: {metrics.box.r.mean():.4f}")