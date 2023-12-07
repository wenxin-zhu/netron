import torchvision.models as models
import torch
import netron
from transformers import GPT2Model, GPT2Config

# # 创建 AlexNet 模型实例
# net = models.AlexNet()
# # 生成虚拟输入数据
# dummy_input = torch.randn(1, 3, 224, 224)
# # 转换模型为 ONNX 格式
# onnx_path = "alexnet_model.onnx"
# torch.onnx.export(net, dummy_input, onnx_path, verbose=True)

# 创建 ResNet-18 模型实例
# net = models.resnet18()
# # 生成虚拟输入数据
# dummy_input = torch.randn(1, 3, 224, 224)
# # 转换模型为 ONNX 格式
# onnx_path = "resnet18_model.onnx"
# torch.onnx.export(net, dummy_input, onnx_path, verbose=True)

# # 创建 MobileNetV2 模型实例
# net = models.mobilenet_v2()
# # 生成虚拟输入数据
# dummy_input = torch.randn(1, 3, 224, 224)
# # 转换模型为 ONNX 格式
# onnx_path = "mobilenetv2_model.onnx"
# torch.onnx.export(net, dummy_input, onnx_path, verbose=True)

# 加载预训练的 GPT-2 模型和配置
model_name = 'gpt2'
config = GPT2Config.from_pretrained(model_name)
gpt2_model = GPT2Model(config)
# 生成虚拟输入数据，假设输入形状为 (batch_size, seq_length)
dummy_input = torch.randint(0, config.vocab_size, (1, 20))
# 转换模型为 ONNX 格式
onnx_path = "gpt2_model.onnx"
torch.onnx.export(gpt2_model, dummy_input, onnx_path, verbose=True)

# 使用netron进行模型可视化
# 模型的路径
modelPath = onnx_path
# 启动模型
netron.start(modelPath)
