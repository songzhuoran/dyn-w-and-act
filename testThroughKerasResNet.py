# encoding=utf8 #
import tensorflow as tf
import numpy as np
import keras
import params
from keras.datasets import cifar10
from keras.optimizers import Adam
import os
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
'''
from keras.backend.tensorflow_backend import set_session  
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  
set_session(tf.Session(config=config))
'''
from quant_util_keras import quantControl

from keras.callbacks import ModelCheckpoint


import resnet
import resnet_un
import resnet_v2
import vggnet
import inception
#import alexnet
from resnet_common import ResNetLoadWeights


'-----------------------------------------------------------image preprocess------------------------------------------------------------'
__ImagePreprocessParam  = dict(
                        #preprocessing_function = vggnet.preprocess_input,
                        validation_split = 0
                    )

__ImageDataFlowParam_val = dict(
                        target_size=(224, 224), 
                        # target_size=(256, 256), 
                        class_mode='categorical', 
                        batch_size=32, 
                        shuffle=False, 
                        );

__ImageDataFlowParam_val_inception = dict(
                        target_size=(299, 299), 
                        # target_size=(256, 256), 
                        class_mode='categorical', 
                        batch_size=32, 
                        shuffle=False, 
                        );

__ImageDataFlowParam_val_alex = dict(
                        target_size=(227, 227), 
                        # target_size=(256, 256), 
                        class_mode='categorical', 
                        batch_size=32, 
                        shuffle=False, 
                        );


img_dg = keras.preprocessing.image.ImageDataGenerator(**__ImagePreprocessParam)



'---------------------------------------------------------------quant utils-------------------------------------------------------------'
def addModelStaticOutput(model, quantControl) :
    ori_inputs = model.inputs
    ori_outputs = model.outputs
    # 在验证时输出统计消息
    listPercentageOutput, listStaticInfoOutput = quantControl.getConvQuantInfo()
    new_outputs = model.outputs + listPercentageOutput + listStaticInfoOutput
    # 重新构造模型
    model = Model(inputs=ori_inputs, outputs=new_outputs)
    return model

def makeModel(
                baseModelBuilder,              
                containStaticOutput = False, 
                listQuant = [],                 # [bool, ...]
                listCntQuantPhase = [],         # [int, ...]
                listGate = [],                  # [[float, ...], ...]
                listfilterGate = [],
                listQuant_input_minmax = [],    # [[int, int, ...], ...]
                listQuant_filter_minmax = [],   # [[int, int], ...]
                listNum_bits = [],              # [[int, ...], ...]
            )  :
    # Set parameter
    quantControl.setQuantParamList(
                            listQuant = listQuant,
                            listCntQuantPhase = listCntQuantPhase,
                            listGate = listGate,
                            listfilterGate = listfilterGate,
                            listQuant_input_minmax = listQuant_input_minmax,
                            listQuant_filter_minmax = listQuant_filter_minmax,
                            listNum_bits = listNum_bits,
                        )
    quantControl.clearConvQuantInfo()
    # Build model
    model = baseModelBuilder()
    # Modify the output
    if containStaticOutput :
        ori_inputs = model.inputs
        ori_outputs = model.outputs
        listPercentageOutput, listStaticInfoOutput = quantControl.getConvQuantInfo()
        new_outputs = model.outputs + listPercentageOutput + listStaticInfoOutput
        model = keras.Model(inputs=ori_inputs, outputs=new_outputs)
    return model

def dealOutput(
                out, 
                n, 
                # labelAns
            ) :
    y = out[0]
    perc_large = np.array(out[1:n+1])                         # (n, cInput)
    stat_large = np.array(out[n+1:n*2+1])                   # (n, cInput, 5) = [imin,imax,fmin,fmax,countH]
    
    perc = np.ceil(np.mean(perc_large, axis = 1))                    # (n, )
    stat_imin_h = np.min(stat_large[:,:,0], axis = 1)       # (n, )
    stat_imax_h = np.ceil(np.max(stat_large[:,:,1], axis = 1))       # (n, )
    stat_imin_l = np.min(stat_large[:,:,2], axis = 1)      # (n, )
    stat_imax_l = np.ceil(np.max(stat_large[:,:,3], axis = 1))      # (n, )
    stat_fmin = np.min(stat_large[:,:,4], axis = 1)         # (n, )
    stat_fmax = np.max(stat_large[:,:,5], axis = 1)         # (n, )
    stat_countH = np.mean(stat_large[:,:,6], axis = 1)       # (n, )
    stat_countA = np.mean(stat_large[:,:,7], axis = 1)  
    stat_countH_1 = np.mean(stat_large[:,:,8], axis = 1)       # (n, )
    stat_countA_1 = np.mean(stat_large[:,:,9], axis = 1)
    
    stat_input_minmax = np.stack([stat_imin_h,stat_imax_h,stat_imin_l,stat_imax_l], axis=-1)
    stat_filter_minmax = np.stack([stat_fmin,stat_fmax], axis=-1)
    
    # acc = np.sum(np.argmax(y, axis = -1) == np.argmax(labelAns, axis = -1)) / y.shape[0]
    # return (y, perc, stat_input_minmax, stat_filter_minmax, acc)
    return (y, perc, stat_input_minmax, stat_filter_minmax, stat_countH, stat_countA, stat_countH_1, stat_countA_1)



'-------------------------------------------------------------------------------------------------------------------------------'
'                                                                                                                               '
'                                                                                                                               '
'                                                         quant config                                                          '
'                                                                                                                               '
'                                                                                                                               '
'-------------------------------------------------------------------------------------------------------------------------------'

alex_quantParam = dict(
                            listQuant = [False] + [False] * 4,
                            listCntQuantPhase = [2] * 5,
                            listGate =  [[0]] * 5,
                            listfilterGate = [0.15] * 5,
                            listQuant_input_minmax =  [[0,0,0,0]] * 5,                                          
                            listQuant_filter_minmax = [[0,0]] * 5,
                            listNum_bits = [[8,4]] * 5,
)


res18_quantParam = dict(
                            listQuant = [False] + [True] * 20,
                            listCntQuantPhase = [2] * 21,
                            listGate =  [[0.06]] * 21, 
                            listfilterGate = [0.17] * 21,
                            listQuant_input_minmax =  params.listQuant_input_minmax_18,                                          
                            listQuant_filter_minmax = params.listQuant_filter_minmax_18,
                            listNum_bits = [[8,4]] * 21,
)


res34_quantParam = dict(
                            listQuant = [False] + [True] * 36,
                            listCntQuantPhase = [2] * 37,
                            listGate =  [[0.06]] * 37, 
                            listfilterGate = [0.16] * 37,
                            listQuant_input_minmax =  params.listQuant_input_minmax_34,                                          
                            listQuant_filter_minmax = params.listQuant_filter_minmax_34,
                            listNum_bits = [[8,4]] * 37,
)


res50_quantParam = dict(
                            listQuant = [False] + [True] * 52,
                            listCntQuantPhase = [2] * 53,
                            listGate = [[0.06]] * 53,   
                            listfilterGate = [0.14] * 53,
                            listQuant_input_minmax =  params.listQuant_input_minmax_50,                                                 
                            listQuant_filter_minmax = params.listQuant_filter_minmax_50,
                            listNum_bits = [[8,4]] * 53,
)


res101_quantParam = dict(
                            listQuant = [False] + [True] * 103,
                            listCntQuantPhase = [2] * 104,
                            listGate = [[0.06]] * 104, 
                            listfilterGate = [0.12] * 104,
                            listQuant_input_minmax =  params.listQuant_input_minmax_101,                                                 
                            listQuant_filter_minmax = params.listQuant_filter_minmax_101,
                            listNum_bits = [[8,4]] * 104,
)


vgg19_quantParam = dict(
                            listQuant = [False] + [True] * 15,
                            listCntQuantPhase = [2] * 16,
                            listGate = [[0.06]] * 16,
                            listfilterGate = [0.08] * 16,
                            listQuant_input_minmax =  params.listQuant_input_minmax_vgg,
                            listQuant_filter_minmax = params.listQuant_filter_minmax_vgg,
                            listNum_bits = [[8,4]] * 16,
)


inceptionv3_quantParam = dict(
                            listQuant = [False] + [True] * 93,
                            listCntQuantPhase = [2] * 94,
                            listGate = [[0.06]] * 94,
                            listfilterGate = [0.18] * 94,
                            listQuant_input_minmax =  params.listQuant_input_minmax_v3,
                            listQuant_filter_minmax = params.listQuant_filter_minmax_v3,
                            listNum_bits = [[8,4]] * 94,
)

makeNetwork = lambda modelFunc : modelFunc(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

dictEval = {
                #'alexnet' :      [makeModel(lambda : makeNetwork(alexnet.AlexNet),  containStaticOutput = False, **alex_quantParam), alexnet.preprocess_input, 8 , __ImageDataFlowParam_val_alex] ,
                'vgg19' :       [makeModel(lambda : makeNetwork(vggnet.VGG19), containStaticOutput = True, **vgg19_quantParam), vggnet.preprocess_input,16, __ImageDataFlowParam_val] ,
                'inception_v3': [makeModel(lambda : makeNetwork(inception.InceptionV3), containStaticOutput = True, **inceptionv3_quantParam), inception.preprocess_input,94, __ImageDataFlowParam_val_inception] ,
                'resnet18' :     [makeModel(lambda : makeNetwork(resnet_un.ResNet18),  containStaticOutput = True, **res18_quantParam), None ,21, __ImageDataFlowParam_val] ,
                'resnet34' :     [makeModel(lambda : makeNetwork(resnet_un.ResNet34),  containStaticOutput = True, **res34_quantParam), None ,37, __ImageDataFlowParam_val] ,
                'resnet50' :    [makeModel(lambda : makeNetwork(resnet.ResNet50),  containStaticOutput = True, **res50_quantParam), resnet.preprocess_input, 53, __ImageDataFlowParam_val] ,
                'resnet101' :   [makeModel(lambda : makeNetwork(resnet.ResNet101), containStaticOutput = True, **res101_quantParam),resnet.preprocess_input, 104, __ImageDataFlowParam_val] ,
            }





def main(models):
    import shelve
    f = open('data.txt', 'w')
    for name in models :

        net, preprocessing_function, deal, __ImageDataFlowParam_val = dictEval[name]   
        img_dg.preprocessing_function = preprocessing_function

        print('------------------------------------------',name,'-----------------------------------------')

        net.load_weights('/home/fubangqi/resnet-imagenet/weights_2/imagenet_' + name + '_mix.h5')       #weight
        

        val_flow = img_dg.flow_from_directory(directory = '/home/benchmark/imagenet/val_centcrop/', **__ImageDataFlowParam_val)
        val_flow.class_mode = 'categorical'
        val_flow.batch_size = 10000
        val_flow.shuffle = True
        val_flow.reset()
        a = next(val_flow)
        out = net.predict(a[0], verbose = 1)
        # y, prec, stat_input_minmax, stat_filter_minmax, acc = dealOutput(out, 53, a[1])
        y, prec, stat_input_minmax, stat_filter_minmax, stat_countH, stat_countA, stat_countH_1, stat_countA_1 = dealOutput(out, deal)
        acc = np.sum(np.argmax(y, axis = -1) == np.argmax(a[1], axis = -1)) / y.shape[0]      
        
        '''
        for i in range(len(prec)//10):
        	print(prec[10*i:10*(i+1)])
        print(prec[len(prec)//10*10:len(prec)//10*10+len(prec)%10])
        #print(prec[48:53])
        print('-----------------')
        '''      
        print('-----------------------------acc----------------------------')
        print(acc)
        print('----------------------------------filter bit per(num of 8bit params out of all params----------------------------------')
        print(  (np.sum(stat_countH)+np.sum(stat_countA))      /       (np.sum(stat_countA)*2)   )
        print('----------------------------------input bit per----------------------------------')
        print(   np.sum(stat_countH_1)    /    np.sum(stat_countA_1)  )
        print('----------------------------------all bit per----------------------------------')
        print(   (np.sum(stat_countH) + np.sum(stat_countA) + np.sum(stat_countH_1))    /    (np.sum(stat_countA)*2 + np.sum(stat_countA_1))  )


if __name__ == '__main__':
    main(models=['vgg19',
                 'inception_v3',
                 'resnet18',
                 'resnet34',
                 'resnet50',
                 'resnet101']

        )