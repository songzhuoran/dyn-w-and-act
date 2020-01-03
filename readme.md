#Readme
## testThroughKerasResNet.py
测试主程序testThroughKerasResNet.py，测试的网络在main(models)函数参数中给出
**可能更改的参数**：
*line160* `res18_quantParam` 网络量化参数
> **listQuant**：每层网络是否量化
**listCntQuantPhase**
**listGate**：每层feature map量化阈值，可以选择统一给出，或者调用对应的`params.listgate`
**listfilterGate**：每层filter量化阈值，统一给出
**listQuant_input_minmax**：每层feature map量化的min、max值，由统计得到，应调用`params.listQuant_input_minmax`
**listQuant_filter_minmax**：每层filter量化的min、max值，由统计得到，应调用`params.listQuant_filter_minmax`
**listNum_bits**：每层量化的bit位，可以为多精度，此时，将listCntQuantPhase改为对应精度数

*line160* `load_weights` 权重载入
> 从`'/home/fubangqi/resnet-imagenet/weights_2/imagenet_' + name + '_mix.h5'`中载入权重，其中name为当前网络名称


## quant_util_keras.py
量化及量化输出方法

##其他.py
网络模型