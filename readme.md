#Readme
## testThroughKerasResNet.py
����������testThroughKerasResNet.py�����Ե�������main(models)���������и���
**���ܸ��ĵĲ���**��
*line160* `res18_quantParam` ������������
> **listQuant**��ÿ�������Ƿ�����
**listCntQuantPhase**
**listGate**��ÿ��feature map������ֵ������ѡ��ͳһ���������ߵ��ö�Ӧ��`params.listgate`
**listfilterGate**��ÿ��filter������ֵ��ͳһ����
**listQuant_input_minmax**��ÿ��feature map������min��maxֵ����ͳ�Ƶõ���Ӧ����`params.listQuant_input_minmax`
**listQuant_filter_minmax**��ÿ��filter������min��maxֵ����ͳ�Ƶõ���Ӧ����`params.listQuant_filter_minmax`
**listNum_bits**��ÿ��������bitλ������Ϊ�ྫ�ȣ���ʱ����listCntQuantPhase��Ϊ��Ӧ������

*line160* `load_weights` Ȩ������
> ��`'/home/fubangqi/resnet-imagenet/weights_2/imagenet_' + name + '_mix.h5'`������Ȩ�أ�����nameΪ��ǰ��������


## quant_util_keras.py
�����������������

##����.py
����ģ��