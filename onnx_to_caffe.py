import onnx
from onnx2caffe.convert import convert_model

if __name__ == "__main__":
    # 机房中你的best.onnx存放路径
    onnx_path = r"./best.onnx"
    # 输出Caffe模型文件
    prototxt_path = r"./best.prototxt"
    caffemodel_path = r"./best.caffemodel"
    # 输入维度：batch=1, RGB3通道, 640×640 匹配训练尺寸
    input_shape = {"images": [1, 3, 640, 640]}

    # 加载ONNX模型
    onnx_model = onnx.load(onnx_path)
    print("已成功加载ONNX模型文件：", onnx_path)

    # 执行ONNX转Caffe转换
    convert_model(
        onnx_model,
        prototxt_path,
        caffemodel_path,
        input_shape=input_shape
    )

    print("======================================")
    print("ONNX转Caffe全部完成！")
    print("网络结构文件prototxt：", prototxt_path)
    print("权重文件caffemodel：", caffemodel_path)