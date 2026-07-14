import os
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# -------------------------- 1、配置PCB缺陷类别 --------------------------
CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# 路径配置
JSON_DIR = r"./dataset/json_label"    # labelme生成json文件夹
VOC_XML_DIR = r"./dataset/voc_xml"    # 中转xml文件夹
YOLO_LABEL_DIR = r"./dataset/labels"
IMG_SRC = r"./dataset/images/all"
TRAIN_IMG = r"./dataset/images/train"
VAL_IMG = r"./dataset/images/val"
TRAIN_LABEL = r"./dataset/labels/train"
VAL_LABEL = r"./dataset/labels/val"

def make_dir(path):
    os.makedirs(path, exist_ok=True)

# 步骤1：Labelme JSON 转 VOC XML
def json2voc():
    make_dir(VOC_XML_DIR)
    for js_file in os.listdir(JSON_DIR):
        if not js_file.endswith(".json"):
            continue
        js_path = os.path.join(JSON_DIR, js_file)
        with open(js_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        img_name = data["imagePath"]
        w = data["imageWidth"]
        h = data["imageHeight"]
        # 构建xml
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = img_name
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(w)
        ET.SubElement(size, "height").text = str(h)
        ET.SubElement(size, "depth").text = "3"
        for shape in data["shapes"]:
            label = shape["label"]
            pts = shape["points"]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = label
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(xmin))
            ET.SubElement(bndbox, "ymin").text = str(int(ymin))
            ET.SubElement(bndbox, "xmax").text = str(int(xmax))
            ET.SubElement(bndbox, "ymax").text = str(int(ymax))
        tree = ET.ElementTree(root)
        xml_save = os.path.join(VOC_XML_DIR, js_file.replace(".json", ".xml"))
        tree.write(xml_save, encoding="utf-8")
    print("=== JSON转XML完成 ===")

# 步骤2：VOC XML 转 YOLO TXT（归一化坐标）
def voc2yolo():
    make_dir(YOLO_LABEL_DIR)
    for xml_file in os.listdir(VOC_XML_DIR):
        if not xml_file.endswith(".xml"):
            continue
        tree = ET.parse(os.path.join(VOC_XML_DIR, xml_file))
        root = tree.getroot()
        size = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)
        txt_content = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            if cls_name not in CLASS_MAP:
                continue
            cls_id = CLASS_MAP[cls_name]
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)
            # YOLO归一化公式
            x_center = (xmin + xmax) / 2 / img_w
            y_center = (ymin + ymax) / 2 / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h
            txt_content.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        txt_path = os.path.join(YOLO_LABEL_DIR, xml_file.replace(".xml", ".txt"))
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_content))
    print("=== XML转YOLO TXT完成 ===")

# 步骤3：数据集划分 训练:验证=8:2
def split_dataset(val_rate=0.2):
    make_dir(TRAIN_IMG)
    make_dir(VAL_IMG)
    make_dir(TRAIN_LABEL)
    make_dir(VAL_LABEL)
    all_imgs = [i for i in os.listdir(IMG_SRC) if i.endswith((".jpg", ".png"))]
    random.seed(42)
    random.shuffle(all_imgs)
    val_num = int(len(all_imgs) * val_rate)
    val_list = all_imgs[:val_num]
    train_list = all_imgs[val_num:]
    # 复制训练集
    for img in train_list:
        shutil.copy(os.path.join(IMG_SRC, img), os.path.join(TRAIN_IMG, img))
        txt_name = Path(img).stem + ".txt"
        shutil.copy(os.path.join(YOLO_LABEL_DIR, txt_name), os.path.join(TRAIN_LABEL, txt_name))
    # 复制验证集
    for img in val_list:
        shutil.copy(os.path.join(IMG_SRC, img), os.path.join(VAL_IMG, img))
        txt_name = Path(img).stem + ".txt"
        shutil.copy(os.path.join(YOLO_LABEL_DIR, txt_name), os.path.join(VAL_LABEL, txt_name))
    print(f"=== 数据集划分完成：训练集{len(train_list)}张，验证集{len(val_list)}张 ===")

if __name__ == "__main__":
    json2voc()
    voc2yolo()
    split_dataset(val_rate=0.2)