import tensorflow as tf
import numpy as np
import keras
import params

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau

'''
from keras.backend.tensorflow_backend import set_session  
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  
set_session(tf.Session(config=config))
'''
from quant_util_keras import quantControl

from keras.callbacks import ModelCheckpoint
'''
import keras_applications
from keras.applications import densenet
from keras.applications import inception_resnet_v2
from keras.applications import inception_v3
from keras.applications import mobilenet
try:
    from keras.applications import mobilenet_v2
except ImportError:
    from keras.applications import mobilenetv2 as mobilenet_v2

from keras.applications import nasnet
from keras.applications import resnet50
from keras.applications import vgg16
from keras.applications import vgg19
from keras.applications import xception
from keras.preprocessing import image
from keras import backend
from keras import layers
from keras import models
from keras import utils


from multiprocessing import Process, Queue


def keras_modules_injection(base_fun):
    
    def wrapper(*args, **kwargs):
        kwargs['backend'] = backend
        kwargs['layers'] = layers
        kwargs['models'] = models
        kwargs['utils'] = utils
        return base_fun(*args, **kwargs)
    return wrapper

for (name, module) in [('resnet', keras_applications.resnet),
                       ('resnet_v2', keras_applications.resnet_v2),
                       ('resnext', keras_applications.resnext)]:
    module.decode_predictions = keras_modules_injection(module.decode_predictions)
    module.preprocess_input = keras_modules_injection(module.preprocess_input)
    for app in dir(module):
        if app[0].isupper():
            setattr(module, app, keras_modules_injection(getattr(module, app)))
    setattr(keras_applications, name, module)
'''
import alexnet
import resnet
import resnet_un
import resnet_v2
import vggnet
import inception
from resnet_common import ResNetLoadWeights

__ImagePreprocessParam  = dict(
                        #preprocessing_function = vggnet.preprocess_input,
                        
                    )

__ImageDataFlowParam_train = dict(
                        target_size=(224, 224), 
                        # target_size=(256, 256), 
                        class_mode='categorical', 
                        batch_size=32, 
                        shuffle=True, 
                        );

__ImageDataFlowParam_train_v3 = dict(
                        target_size=(299, 299),  
                        # target_size=(256, 256), 
                        class_mode='categorical', 
                        batch_size=32, 
                        shuffle=True, 
                        );

__ImageDataFlowParam_train_alex = dict(
                        target_size=(227, 227), 
                        # target_size=(256, 256), 
                        class_mode='categorical', 
                        batch_size=32, 
                        shuffle=True, 
                        );



__ImageDataFlowParam_val = dict(
                        target_size=(224, 224), 
                        # target_size=(256, 256), 
                        class_mode='categorical', 
                        batch_size=32, 
                        shuffle=False, 
                        );

__ImageDataFlowParam_val_v3 = dict(
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
#val_flow = img_dg.flow_from_directory(directory = '/home/benchmark/imagenet/val_centcrop/', **__ImageDataFlowParam_val)
#train_flow = img_dg.flow_from_directory(directory = '/home/benchmark/imagenet/train/', **__ImageDataFlowParam_train)
#test_flow=img_dg.flow_from_directory(directory = '/home/benchmark/imagenet/test', **__ImageDataFlowParam_val)




# res101 = keras_applications.resnet.ResNet101(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# res101.compile(opt, loss = ['categorical_crossentropy'], metrics = ['accuracy'])

# res152 = keras_applications.resnet.ResNet152(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# res152.compile(opt, loss = ['categorical_crossentropy'], metrics = ['accuracy'])

# res152v2 = keras_applications.resnet_v2.ResNet152V2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

makeNetwork = lambda modelFunc : modelFunc(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

'''
From quantControl add static info to model outputs
'''
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
                baseModelBuilder,               # A function to build and return a keras model, i.e. lambda : makeNetwork(keras_applications.resnet.ResNet50)
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
    
    stat_input_minmax = np.stack([stat_imin_h,stat_imax_h,stat_imin_l,stat_imax_l], axis=-1)
    stat_filter_minmax = np.stack([stat_fmin,stat_fmax], axis=-1)
    
    # acc = np.sum(np.argmax(y, axis = -1) == np.argmax(labelAns, axis = -1)) / y.shape[0]
    
    # return (y, perc, stat_input_minmax, stat_filter_minmax, acc)
    return (y, perc, stat_input_minmax, stat_filter_minmax, stat_countH, stat_countA)

'''
dictEval = {
                'res50' :       [makeNetwork(keras_applications.resnet.ResNet50),       keras_applications.resnet.preprocess_input] ,
                'res101' :      [makeNetwork(keras_applications.resnet.ResNet101),      keras_applications.resnet.preprocess_input] ,
                'res152' :      [makeNetwork(keras_applications.resnet.ResNet152),      keras_applications.resnet.preprocess_input] ,
                'res50v2' :     [makeNetwork(keras_applications.resnet_v2.ResNet50V2),  keras_applications.resnet_v2.preprocess_input] ,
                'res101v2' :    [makeNetwork(keras_applications.resnet_v2.ResNet101V2), keras_applications.resnet_v2.preprocess_input] ,
                'res152v2' :    [makeNetwork(keras_applications.resnet_v2.ResNet152V2), keras_applications.resnet_v2.preprocess_input] ,
            }
'''
'''
dictEval = {
                'resnet50' :       [makeNetwork(resnet.ResNet50),       resnet.preprocess_input] ,
                'resnet101' :      [makeNetwork(resnet.ResNet101),      resnet.preprocess_input] ,
                'resnet152' :      [makeNetwork(resnet.ResNet152),      resnet.preprocess_input] ,
                'resnet50v2' :     [makeNetwork(resnet_v2.ResNet50V2),  resnet_v2.preprocess_input] ,
                'resnet101v2' :    [makeNetwork(resnet_v2.ResNet101V2), resnet_v2.preprocess_input] ,
                'resnet152v2' :    [makeNetwork(resnet_v2.ResNet152V2), resnet_v2.preprocess_input] ,
            }
'''
# run(listGate):

alex_quantParam = dict(
                            listQuant = [False] + [False] * 4,
                            listCntQuantPhase = [2] * 5,
                            listGate =  [[0]] * 5,
                            listfilterGate = [0.15] * 5,
                            listQuant_input_minmax =  [[0,0,0,0]] * 5,                                          
                            listQuant_filter_minmax = [[0,0]] * 5,
                            listNum_bits = [[8,4]] * 5,
)

'''
res18_quantParam = dict(
                            listQuant = [False] + [True] * 20,
                            listCntQuantPhase = [2] * 21,
                            listGate=  params.listGate_18,
                            listQuant_input_minmax =  params.listQuant_input_minmax_18,                                          
                            listQuant_filter_minmax = params.listQuant_filter_minmax_18,
                            listNum_bits = [[8,4]] * 21,
)
'''

res18_quantParam = dict(
                            listQuant = [False] + [True] * 20,
                            listCntQuantPhase = [2] * 21,
                            listGate =  [[0.06]] * 21, 
                            listfilterGate = [0.18] * 21,
                            listQuant_input_minmax =  params.listQuant_input_minmax_18,                                          
                            listQuant_filter_minmax = params.listQuant_filter_minmax_18,
                            listNum_bits = [[8,4]] * 21,
)

'''
res34_quantParam = dict(
                            listQuant = [False] + [True] * 36,
                            listCntQuantPhase = [2] * 37,
                            listGate=  params.listGate_34, 
                            listQuant_input_minmax =  params.listQuant_input_minmax_34,                                          
                            listQuant_filter_minmax = params.listQuant_filter_minmax_34,
                            listNum_bits = [[8,4]] * 37,
)
'''

res34_quantParam = dict(
                            listQuant = [False] + [True] * 36,
                            listCntQuantPhase = [2] * 37,
                            listGate =  [[0.06]] * 37, 
                            listfilterGate = [0.16] * 37,
                            listQuant_input_minmax =  params.listQuant_input_minmax_34,                                          
                            listQuant_filter_minmax = params.listQuant_filter_minmax_34,
                            listNum_bits = [[8,4]] * 37,
)

'''
res50_quantParam = dict(
                            listQuant = [False] + [True] * 52,
                            listCntQuantPhase = [2] * 53,
                            listGate=params.listGate_50,   
                            listQuant_input_minmax =  params.listQuant_input_minmax_50,                                                 
                            listQuant_filter_minmax = params.listQuant_filter_minmax_50,
                            listNum_bits = [[8,4]] * 53,
)
'''
res50_quantParam = dict(
                            listQuant = [False] + [True] * 52,
                            listCntQuantPhase = [2] * 53,
                            listGate = [[0.06]] * 53,   
                            listfilterGate = [0.14] * 53,
                            listQuant_input_minmax =  params.listQuant_input_minmax_50,                                                 
                            listQuant_filter_minmax = params.listQuant_filter_minmax_50,
                            listNum_bits = [[8,4]] * 53,
)

'''
res101_quantParam = dict(
                            listQuant = [False] + [False] * 103,
                            listCntQuantPhase = [2] * 104,
                            listGate=params.listGate_101,   
                            listQuant_input_minmax =  params.listQuant_input_minmax_101,                                                 
                            listQuant_filter_minmax = params.listQuant_filter_minmax_101,
                            listNum_bits = [[8,4]] * 104,
)
'''

res101_quantParam = dict(
                            listQuant = [False] + [True] * 103,
                            listCntQuantPhase = [2] * 104,
                            listGate = [[0.06]] * 104, 
                            listfilterGate = [0.12] * 104,
                            listQuant_input_minmax =  params.listQuant_input_minmax_101,                                                 
                            listQuant_filter_minmax = params.listQuant_filter_minmax_101,
                            listNum_bits = [[8,4]] * 104,
)

'''
vgg19_quantParam = dict(
                            listQuant = [False] + [True] * 15,
                            listCntQuantPhase = [2] * 16,
                            listGate= params.listGate_vgg,
                            listQuant_input_minmax =  params.listQuant_input_minmax_vgg,
                            listQuant_filter_minmax = params.listQuant_filter_minmax_vgg,
                            listNum_bits = [[8,4]] * 16,
)
'''

vgg19_quantParam = dict(
                            listQuant = [False] + [True] * 15,
                            listCntQuantPhase = [2] * 16,
                            listGate = [[0.06]] * 16,
                            listfilterGate = [0.08] * 16,
                            listQuant_input_minmax =  params.listQuant_input_minmax_vgg,
                            listQuant_filter_minmax = params.listQuant_filter_minmax_vgg,
                            listNum_bits = [[8,4]] * 16,
)

'''
inceptionv3_quantParam = dict(
                            listQuant = [False] + [False] * 93,
                            listCntQuantPhase = [2] * 94,
                            listGate= params.listGate_v3,
                            listQuant_input_minmax =  params.listQuant_input_minmax_v3,
                            listQuant_filter_minmax = params.listQuant_filter_minmax_v3,
                            listNum_bits = [[8,4]] * 94,
)
'''

inceptionv3_quantParam = dict(
                            listQuant = [False] + [True] * 93,
                            listCntQuantPhase = [2] * 94,
                            listGate = [[0.06]] * 94,
                            listfilterGate = [0.18] * 94,
                            listQuant_input_minmax =  params.listQuant_input_minmax_v3,
                            listQuant_filter_minmax = params.listQuant_filter_minmax_v3,
                            listNum_bits = [[8,4]] * 94,
)



dictEval = {
                #'alexnet' :      [makeModel(lambda : makeNetwork(alexnet.AlexNet),  containStaticOutput = False, **alex_quantParam), alexnet.preprocess_input, 8 , __ImageDataFlowParam_val_alex, __ImageDataFlowParam_train_alex] ,
                'vgg19' :       [makeModel(lambda : makeNetwork(vggnet.VGG19), containStaticOutput = False, **vgg19_quantParam), vggnet.preprocess_input,16, __ImageDataFlowParam_val, __ImageDataFlowParam_train] ,
                
                'resnet18' :    [makeModel(lambda : makeNetwork(resnet_un.ResNet18),  containStaticOutput = False, **res18_quantParam), None ,21, __ImageDataFlowParam_val, __ImageDataFlowParam_train] ,
                #'resnet34' :     [makeModel(lambda : makeNetwork(resnet_un.ResNet34),  containStaticOutput = False, **res34_quantParam), None ,37, __ImageDataFlowParam_val, __ImageDataFlowParam_train] ,
                #'inception_v3' :[makeModel(lambda : makeNetwork(inception.InceptionV3), containStaticOutput = False, **inceptionv3_quantParam), inception.preprocess_input,94, __ImageDataFlowParam_val_v3, __ImageDataFlowParam_train_v3] ,
                #'resnet50' :    [makeModel(lambda : makeNetwork(resnet.ResNet50),  containStaticOutput = False, **res50_quantParam), resnet.preprocess_input,53, __ImageDataFlowParam_val, __ImageDataFlowParam_train] ,
                #'resnet101' :   [makeModel(lambda : makeNetwork(resnet.ResNet101), containStaticOutput = False, **res101_quantParam),resnet.preprocess_input,104, __ImageDataFlowParam_val, __ImageDataFlowParam_train] ,
            }

'''
dictEval = {
                'resnet50' :       [makeModel(lambda : makeNetwork(resnet.ResNet50), containStaticOutput = False, **res50_quantParam), resnet.preprocess_input] ,
            }

dictEval = {
                'resnet18' :       [makeModel(lambda : makeNetwork(resnet.ResNet18), containStaticOutput = False, **res18_quantParam), resnet.preprocess_input] ,
            }
'''
'''
修改next方法以实现crop
ResNet的预处理方法是先把图像放缩至(256,256),之后中心crop(224,224)
'''
'''
# TODO: 搞优雅一点
val_flow.ori_next = val_flow.next
def doCentCrop() :
    t = val_flow.ori_next()
    return (t[0][:,16:(16+224),16:(16+224),:], t[1])

val_flow.next = doCentCrop
'''
def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-7
    if epoch > 3:
        lr *= 1e-1
    elif epoch > 2:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

import shelve

for name in dictEval.keys() :
    net, preprocessing_function, deal, __ImageDataFlowParam_val, __ImageDataFlowParam_train = dictEval[name]
    opt = keras.optimizers.adam(lr =lr_schedule(0))
    img_dg.preprocessing_function = preprocessing_function

    val_flow = img_dg.flow_from_directory(directory = '/home/benchmark/imagenet/val_centcrop/', **__ImageDataFlowParam_val)
    train_flow = img_dg.flow_from_directory(directory = '/home/benchmark/imagenet/train/', **__ImageDataFlowParam_train)


    print('------------------------------------------',name,'-----------------------------------------')
    net.compile(opt, loss = ['categorical_crossentropy'] , metrics = ['accuracy','top_k_categorical_accuracy'] )


    net.load_weights('/home/fubangqi/resnet-imagenet/weights_2/imagenet_' + name + '_mix.h5')
    #ResNetLoadWeights(net, name, 'imagenet', include_top = True)

    #net.compile(opt, loss = ['categorical_crossentropy'], metrics = ['accuracy'])
    
    
    
    '''
    data = net.evaluate_generator(val_flow, steps = len(val_flow), verbose = 1)
    print('loss: ',data[0])
    print('acc: ',data[1])
    '''


    #'''



    checkpoint = ModelCheckpoint('/home/fubangqi/resnet-imagenet/weights_2/imagenet_' + name + '_mix_1.h5', 
        monitor='val_acc', save_best_only=True, mode='max')



    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    callbacks = [checkpoint, lr_reducer, lr_scheduler]


    net.fit_generator(train_flow, validation_data=val_flow, steps_per_epoch=len(train_flow)//32, epochs=4, callbacks=[checkpoint])





    data = net.evaluate_generator(val_flow, steps = len(val_flow), verbose = 1)
    print('loss: ',data[0])
    print('acc: ',data[1])






    #'''

  

    '''
    net.save('/home/fubangqi/resnet-imagenet/weights/imagenet_quant_1_model.h5')
    
    eval_hist = net.evaluate_generator(val_flow, steps = len(val_flow), verbose = 1)
    print(eval_hist)
    val_flow.class_mode = 'none'
    #out = net.predict_generator(val_flow, steps = len(val_flow), verbose = 1)
    # out = net.predict_generator(val_flow, steps = 1, verbose = 1)
    #y, prec, stat_input_minmax, stat_filter_minmax = dealOutput(out, 53)
    '''
    
    '''
    val_flow.class_mode = 'categorical'
    val_flow.batch_size = 8192
    val_flow.shuffle = True
    val_flow.reset()
    a = next(val_flow)
    out = net.predict(a[0], verbose = 1)
    
    # y, prec, stat_input_minmax, stat_filter_minmax, acc = dealOutput(out, 53, a[1])
    y, prec, stat_input_minmax, stat_filter_minmax, stat_countH = dealOutput(out, 53)
    acc = np.sum(np.argmax(y, axis = -1) == np.argmax(a[1], axis = -1)) / y.shape[0]
    '''
    
    # print(name, eval_hist)
    '''
    for i in range(8):
        print(prec[6*i:6*i+6])
    print(prec[48:53])
    print(acc)
    '''
    #print(stat_input_minmax[0:25])
    '''
    for i in range(53):
        print('[',end='  ')
        for j in range(4):
            print(stat_input_minmax[i][j],end=' '),print(' , ',end=' ')
        print('],')
    print(" ")
    '''
    #print(stat_input_minmax[25:,])
    '''
    db = shelve.open('test_crop_quant')
    db['eval_hist_'+name] = dict(y=y, prec=prec, stat_input_minmax=stat_input_minmax, stat_filter_minmax=stat_filter_minmax, stat_countH=stat_countH, acc=acc)
    db.close()
    '''
    #quit()


'''
import shelve

db = shelve.open('test0')

for key in db.keys() :
'''

