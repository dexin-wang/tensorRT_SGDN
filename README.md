# tensorRT_SGDN
Use tensorRT to deploy a grasp detection network (SGDN)

## 1. TensorRT简介
tensorRT是可以在NVIDIA的各种GP U下运行的一个**C++推理框架**。我们使用Pytorch、TF训练好的模型，可以转化为TensorRT的格式，然后用TensorRT推理引擎去运行这个模型，从而提高模型在英伟达GPU上的运行速度，一般可提高几倍~几十倍。

主流的pytorch部署路径：
- pytorch $\rightarrow$ ONNX $\rightarrow$ TensorRT
- [torch2trt](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA-AI-IOT/torch2trt)
- [torch2trt_dynamic](https://link.zhihu.com/?target=https%3A//github.com/grimoire/torch2trt_dynamic)
- [TRTorch](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/TRTorch)

## 2. 抓取检测部署流程
此处使用的抓取检测网络来自于我先前投稿的论文：High-performance Pixel-level Grasp Detection based on Adaptive Grasping and Grasp-aware Network。该方法在康奈尔数据集上取得了99.09%的抓取检测精度，并在实际多物体堆叠场景中取得了95.71%的抓取成功率，实验演示视频在youtube上：https://www.youtube.com/watch?v=KUa3XlVwDsU。论文下载地址：https://www.techrxiv.org/articles/preprint/High-performance_Pixel-level_Grasp_Detection_based_on_Adaptive_Grasping_and_Grasp-aware_Network/14680455

部署TensorRT需要安装pytorch、tensorRT、ONNX等依赖，具体安装方法网上都比较详细，这里只展示我采用的版本信息：
```powershell
ubuntu:           16.06
TensorRT:         7.0.0
ONNX IR version:  0.0.4
Opset version:    10
Producer name:    pytorch 1.2.0
GPU:              TITAN Xp
CUDA:             10.0
Driver Version:   430.14
```
可以使用python或者C++进行部署，这里我采用C++。

### 2.1 将pytorch的网络生成onnx文件
pytorch提供了生成onnx模型的方法，代码如下：
```python
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("model_path") # pytorch模型加载

model.eval()
x = torch.randn((1, 3, 320, 320))   # 生成张量
x = x.to(device)
torch.onnx.export(model,
                  x,
                  "ckpt/sgdn.onnx",
                  verbose =True,
                  opset_version=10,
                  do_constant_folding=True,	# 是否执行常量折叠优化
                  input_names=["input"],	# 输入名
                  output_names=["output_able", "output_angle", "output_width"])	# 输出名
```
在生成的时候，有一点需要注意，不要在网络中使用插值上采样，否则在tensorRT推理会报错，使用`torch.nn.UpsamplingNearest2d()`代替插值上采样。关于这个问题的讨论：`https://github.com/NVIDIA/TensorRT/issues/284`。

onnx文件可以在我的谷歌网盘下载：
```powershell
https://drive.google.com/file/d/1AGyjRTWIw85ctwP6VsBDCmR0mE8NdRLu/view?usp=sharing
```
使用python的onnx包检查sgdn.onnx是否有效，程序如下：
```python
import onnx
model_path = 'sgdn.onnx'
# 验证模型合法性
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
```
使用netron工具查看网络架构以及网络的输入输出形状，在线的netron网址如下：
```
https://netron.app/
```
网页截图如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021070114373333.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMDgxMjA4,size_16,color_FFFFFF,t_70#pic_center)


### 2.2 生成txt格式的图像数据
正常来说，要使用opencv读取图像，或者通过ROS系统订阅图像，这里为了方便测试，事先将图像转化为txt格式。由于抓取检测网络的输入尺寸为$(batch,3,320,320)$，所以首先裁剪图像中间的$(320,320)$区域，然后将像素值保存至txt文件。每行存储320个值，共320*3行，其中，前320行为B通道，后面依次为G和R通道。
程序在github下载：
```powershell
https://github.com/dexin-wang/tensorRT_SGDN/tree/main/create_txt
```
通过`python3 create_txt.py`生成txt文件。

### 2.3 TensorRT推理
因为还在测试阶段，所以我的C++程序是在TensorRT官方提供的示例代码上改的，在我的github上可以下载：
```powershell
https://github.com/dexin-wang/tensorRT_SGDN/tree/main/sampleOnnxSGDN
```
按照网上的教程安装好TensorRT后，将上面链接中的`sampleOnnxSGDN`文件夹放到`/home/.../TensorRT-7.0.0.11/samples/`中，并在`/home/.../TensorRT-7.0.0.11/samples/Makefile`文件的第`39`行中，增加一项`sampleOnnxSGDN`：
```
samples=... sampleOnnxSGDN ...
```
然后，将下载的`sgdn.onnx`放在`/home/.../TensorRT-7.0.0.11/samples/sampleOnnxSGDN/data/`文件夹下。另外，可能需要修改程序中涉及的文件路径。
这样，就可以进行编译和运行了。
#### 编译
```powershell
cd /home/.../TensorRT-7.0.0.11/samples/sampleOnnxSGDN/
make
```
编译完成后，在`/home/.../TensorRT-7.0.0.11/bin`路径下，生成了两个文件：
```
sample_onnx_sgdn
sample_onnx_sgdn_debug
```
#### 运行
```powershell
cd /home/.../TensorRT-7.0.0.11/bin
./sample_onnx_sgdn
```
如果以下顺利的话，可以看到运行结果:
```powershell
[07/01/2021-10:14:08] [I] Building and running a GPU inference engine for Onnx MNIST
----------------------------------------------------------------
Input filename:   /home/wangdx/tensorRT/TensorRT-7.0.0.11/samples/sampleOnnxSGDN/data/sgdn.onnx
ONNX IR version:  0.0.4
Opset version:    10
Producer name:    pytorch
Producer version: 1.2
Domain:
Model version:    0
Doc string:
----------------------------------------------------------------
[07/01/2021-10:14:13] [I] [TRT] Some tactics do not have sufficient workspace memory to run. Increa
[07/01/2021-10:14:51] [I] [TRT] Detected 1 inputs and 3 output network tensors.
[07/01/2021-10:14:51] [W] [TRT] Current optimization profile is: 0. Please ensure there are no enqu
[07/01/2021-10:14:51] [I] Output:
[07/01/2021-10:14:51] [I] (row, col) = 233, 187
confidence = 0.996655
&&&& PASSED TensorRT.sample_onnx_sgdn # ./sample_onnx_sgdn
```
结果表示，在$(233,187)$位置处的抓取点置信度最高，置信度为0.996655。由于一开始裁剪了图像中间的$(320,320)$，所以在原图中，预测的抓取点的位置是$(233+80,187+160)=(313,347)$。代码中没有给出解析抓取角和抓取宽度的代码，后续我会更新代码并发布。

## 3. 报错总结
**报错1**：

onnx->tensorRT时

```shell
While parsing node number 360 [Resize]:
ERROR: ModelImporter.cpp:124 In function parseGraph:
[5] Assertion failed: ctx->tensors().count(inputName)
```

解决：不要使用双线性插值，使用 nn.UpsamplingNearest2d((h,w))

**报错2**：

```shell
[06/30/2021-17:22:19] [E] [TRT] Network has dynamic or shape inputs, but no optimization profile has been defined.
[06/30/2021-17:22:19] [E] [TRT] Network validation failed.
&&&& FAILED TensorRT.sample_onnx_sgdn # ./sample_onnx_sgdn
```

解决：在生成将pytorch转为onnx时，不要设置dynamic_axes。检测是否正确的方法：在netron中网络的输入shape是$(1,3,320,320)$。而不是$(batch\_size,3,320,320)$。

**报错3**：

```shell
Some tactics do not have sufficient workspace memory to run. Increasing workspace size may increase performance, please check verbose output.
[06/30/2021-17:52:05] [I] [TRT] Detected 1 inputs and 3 output network tensors.
[06/30/2021-17:52:06] [W] [TRT] Current optimization profile is: 0. Please ensure there are no enqueued operations pending in this context prior to switching profiles 
Segmentation fault (core dumped)
```
解决：读取二进制的文件错误，改成了读取txt。


## 4. 参考
https://zhuanlan.zhihu.com/p/371239130

https://zhuanlan.zhihu.com/p/348301573
