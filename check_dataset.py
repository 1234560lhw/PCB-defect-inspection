import os

# 路径配置
train_img_dir = r"./dataset/images/train"
train_label_dir = r"./dataset/labels/train"
val_img_dir = r"./dataset/images/val"
val_label_dir = r"./dataset/labels/val"

def check_match(img_path, label_path):
    # 获取所有图片纯文件名
    img_names = set()
    for f in os.listdir(img_path):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            stem = os.path.splitext(f)[0]
            img_names.add(stem)
    # 获取所有标签纯文件名
    label_names = set()
    for f in os.listdir(label_path):
        if f.endswith(".txt"):
            stem = os.path.splitext(f)[0]
            label_names.add(stem)

    img_only = img_names - label_names  # 有图无标签
    label_only = label_names - img_names  # 有标签无图

    print(f"====== {img_path} 校验结果 ======")
    print(f"图片总数：{len(img_names)}")
    print(f"标签总数：{len(label_names)}")
    print(f"只有图片、缺少txt标签：{len(img_only)}")
    if img_only:
        print("缺失标签的图片名列表：", sorted(list(img_only)))
    print(f"只有txt标签、缺少图片：{len(label_only)}")
    if label_only:
        print("缺失图片的标签名列表：", sorted(list(label_only)))
    print("-" * 50)

if __name__ == "__main__":
    check_match(train_img_dir, train_label_dir)
    check_match(val_img_dir, val_label_dir)