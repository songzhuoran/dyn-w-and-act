#-- coding:UTF-8 --
import tensorflow as tf
import numpy as np
import keras

array_ops = tf
sparse_ops = tf
sparse_tensor = tf
sparse_ops = tf

py_all = all

# from keras.backend, see https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/python/keras/backend.py
def is_sparse(tensor):
    """Returns whether a tensor is a sparse tensor.
    Arguments:
      tensor: A tensor instance.
    Returns:
      A boolean.
    Example:
    ```python
      >>> from keras import backend as K
      >>> a = K.placeholder((2, 2), sparse=False)
      >>> print(K.is_sparse(a))
      False
      >>> b = K.placeholder((2, 2), sparse=True)
      >>> print(K.is_sparse(b))
      True
    ```
    """
    return isinstance(tensor, sparse_tensor.SparseTensor)

# from keras.backend, see https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/python/keras/backend.py
def to_dense(tensor):
    """Converts a sparse tensor into a dense tensor and returns it.
    Arguments:
      tensor: A tensor instance (potentially sparse).
    Returns:
      A dense tensor.
    Examples:
    ```python
      >>> from keras import backend as K
      >>> b = K.placeholder((2, 2), sparse=True)
      >>> print(K.is_sparse(b))
      True
      >>> c = K.to_dense(b)
      >>> print(K.is_sparse(c))
      False
    ```
    """
    if is_sparse(tensor):
        return sparse_ops.sparse_tensor_to_dense(tensor)
    else:
        return tensor

# from keras.backend, see https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/python/keras/backend.py
def concatenate(tensors, axis=-1):
    """Concatenates a list of tensors alongside the specified axis.
    Arguments:
    tensors: list of tensors to concatenate.
    axis: concatenation axis.
    Returns:
    A tensor.
    """
    if axis < 0:
        rank = ndim(tensors[0])
        if rank:
            axis %= rank
        else:
            axis = 0
    
    if py_all(is_sparse(x) for x in tensors):
        return sparse_ops.sparse_concat(axis, tensors)
    else:
        return array_ops.concat([to_dense(x) for x in tensors], axis)

# from keras.backend, see https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/python/keras/backend.py
# OR, use keras.backend.repeat_elements instead
def repeat_elements(x, rep, axis):
    """Repeats the elements of a tensor along an axis, like `np.repeat`.
    If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
    will have shape `(s1, s2 * rep, s3)`.
    Arguments:
      x: Tensor or variable.
      rep: Python integer, number of times to repeat.
      axis: Axis along which to repeat.
    Returns:
      A tensor.
    """
    x_shape = x.shape.as_list()
    # For static axis
    if x_shape[axis] is not None:
        # slices along the repeat axis
        splits = array_ops.split(  value=x,
                            num_or_size_splits=x_shape[axis],
                            axis=axis)
        # repeat each slice the given number of reps
        x_rep = [s for s in splits for _ in range(rep)]
        return concatenate(x_rep, axis)
    
    # Here we use tf.tile to mimic behavior of np.repeat so that
    # we can handle dynamic shapes (that include None).
    # To do that, we need an auxiliary axis to repeat elements along
    # it and then merge them along the desired axis.
    
    # Repeating
    auxiliary_axis = axis + 1
    x_shape = array_ops.shape(x)
    x_rep = array_ops.expand_dims(x, axis=auxiliary_axis)
    reps = np.ones(len(x.shape) + 1)
    reps[auxiliary_axis] = rep
    x_rep = array_ops.tile(x_rep, reps)
    
    # Merging
    reps = np.delete(reps, auxiliary_axis)
    reps[axis] = rep
    reps = array_ops.constant(reps, dtype='int32')
    x_shape *= reps
    x_rep = array_ops.reshape(x_rep, x_shape)
    
    # Fix shape representation
    x_shape = x.shape.as_list()
    x_rep.set_shape(x_shape)
    x_rep._keras_shape = tuple(x_shape)
    return x_rep

def expand2D(tensor, stride_num) :
    '''
    tensor_e1 = repeat_elements(tensor, stride_num, 1)          # or use keras.backend.repeat_elements instead
    tensor_e2 = repeat_elements(tensor_e1, stride_num, 2)
    '''
    tensor_e1 = keras.backend.repeat_elements(tensor, stride_num, 1)          # use keras.backend.repeat_elements instead
    tensor_e2 = keras.backend.repeat_elements(tensor_e1, stride_num, 2)
    return tensor_e2

__SOFT_GREATER_USE_SOFT_FONT = False

'''
@tf.custom_gradient
def cg_soft_greater(x, gate) :
#     ridge = 1
    ridge = 5
    # font = tf.cast(x > gate, tf.float32)
    font = tf.sigmoid((x - gate) * ridge) if __SOFT_GREATER_USE_SOFT_FONT else tf.cast(x > gate, tf.float32)
    back = tf.sigmoid((x - gate) * ridge)
    def grad(dy):
        # print('grad: dy =',dy)
        ret_grad = tf.gradients(back, [x, gate], grad_ys=dy)
        # print('grad: return grad =',ret_grad)
        return ret_grad
    return font, grad
'''

@tf.custom_gradient
def cg_soft_greater(x, gate) :
#     ridge = 1
    ridge = 5
    # font = tf.cast(x > gate, tf.float32)
    font = tf.sigmoid((x - gate) * ridge) if __SOFT_GREATER_USE_SOFT_FONT else tf.cast(x >= gate, tf.float32)
    back = tf.sigmoid((x - gate) * ridge)
    def grad(dy):
        # print('grad: dy =',dy)
        ret_grad = tf.gradients(back, [x, gate], grad_ys=dy)
        # print('grad: return grad =',ret_grad)
        return ret_grad
    return font, grad

'''
def buildConv2DLayer(input, filter, gate=0.25, quant_input_minmax = [0,0], quant_filter_minmax = [0,0], num_bits_h=4, num_bits_l=2,
                    strides = [1,1], padding = 'SAME', dilations = [1,1], quant=True, useQuantMinMaxVars = True) :
'''

def buildConv2DLayer(   input,
                        channels,
                        filter,                         # 具体的卷积核Variable
                        strides = [1,1],
                        padding = 'SAME',
                        dilations = [1,1],
                        quant = True,
                        cntQuantPhase = 2,              # TODO: 目前仅支持2
                        gate = [0.25],                    # gate 应该按从高到低的顺序排列，对应精度从大到小
                        filterGate = 0.15,
                        # TODO: quant_input_minmax没有实现对多层的支持，max_l = max_h / 2
                        quant_input_minmax = [0,0,0,0],     # [min1, max1, min2, max2, ...]
                        quant_filter_minmax = [0,0],    # [min, max]
                        num_bits = [8,4],               # num_bits应该按精度从高到低的顺序排列
                        useQuantMinMaxVars = True,
                        printDebug = True
                    ) :
    
    # s=int(input.get_shape()[0])       # 不要调用get_shape;否则不能使用(None, 32, 32, 3)定义输入
    # h=int(input.get_shape()[1])
    # w=int(input.get_shape()[2])
    # c=int(input.get_shape()[3])
    shape = input.shape.as_list()
    
    s = shape[0] if shape[0] is not None else tf.shape(input)[0]
    h = shape[1] if shape[1] is not None else tf.shape(input)[1]
    w = shape[2] if shape[2] is not None else tf.shape(input)[2]
    c = shape[3] if shape[3] is not None else tf.shape(input)[3]
    
    if padding=='SAME' :
        h_o=h
        w_o=h
    else:
        h_o=h-2
        w_o=w-2
    
    if (isinstance(w, tf.Tensor)) :
        # 即shape[2] is None, 没有在构造阶段定义输入尺寸
        stride_num=8
    else :	
        if w>=32:
            stride_num=1
        elif w>=16 and w<32:
            stride_num=1
        else:
            stride_num=1
    
    filter_one=tf.ones([stride_num,stride_num,c,1],dtype = 'float32')
    mask_one=tf.ones([s,h,w,c],dtype = 'float32')
    with tf.variable_scope('conv_predicter'):
        gateConv=tf.nn.depthwise_conv2d(
            input, 
            filter_one, 
            ([1,stride_num,stride_num,1]), 
            padding='SAME', 
            rate=[1,1], 
        )
    
    
    
    
    input_min_h=quant_input_minmax[0]
    input_max_h=quant_input_minmax[1]
    input_min_l=quant_input_minmax[2]
    input_max_l=quant_input_minmax[3]
    filter_min=quant_filter_minmax[0]
    filter_max=quant_filter_minmax[1]
    #print('buildConv2DLayer, quant =', quant, ', quant_input_minmax =', quant_input_minmax, ', quant_filter_minmax =', quant_filter_minmax, ', gate =', gate,', num_bits_[h,l] =', num_bits) if printDebug else None
    
    # 为了支持min,max作为tf的变量(以自动训练),将量化函数修改为fake_quant_with_min_max_vars
    # 注意fake_quant_with_min_max_vars的自动求导,min,max导数只反映了超出量化区间的输入数量
    if useQuantMinMaxVars :
        fakeQuantOp = tf.fake_quant_with_min_max_vars   # fake_quant_with_min_max_vars也支持min,max输入常数
    else :
        fakeQuantOp = tf.fake_quant_with_min_max_args
    
    input_h=fakeQuantOp(
                    input, 
                    min = input_min_h,
                    max = input_max_h,
                    num_bits = num_bits[0],
                )
    input_l=fakeQuantOp(
                    input, 
                    min = input_min_l,
                    max = input_max_l,
                    num_bits = num_bits[1],
                )
    filter_h=fakeQuantOp(
                    filter, 
                    min = filter_min,
                    max = filter_max,
                    num_bits =num_bits[0],
                )
    filter_l=fakeQuantOp(
                    filter, 
                    min = filter_min,
                    max = filter_max,
                    num_bits = num_bits[1],
                )
    
    maxGateConv=tf.reduce_max(gateConv)
    minGateConv=tf.reduce_min(gateConv)
    
    actualGateValue = gate[0] * maxGateConv
    
    '''
    one = tf.ones_like(gateConv,dtype = 'float32')
    zero = tf.zeros_like(gateConv,dtype = 'float32')
    
    with tf.device('/cpu:0'):
        # points = tf.where(gateConv > gate, x=one, y=zero)
        points = tf.where(gateConv > actualGateValue, x=one, y=zero)
    '''
    points = cg_soft_greater(gateConv, actualGateValue)
    
    # '''
    def expand2D(tensor, stride_num) :
        mask_h=tensor.get_shape()[1]
        mask_w=tensor.get_shape()[2]
        points_tr=tf.transpose(tensor,[2,0,1,3])
        mask_col=tf.gather(points_tr,sorted(stride_num*[i for i in range(mask_w)])[0:w])
        mask_tr=tf.transpose(mask_col,[2,1,0,3])
        mask_raw=tf.gather(mask_tr,sorted(stride_num*[i for i in range(mask_h)])[0:h])
        mask=tf.transpose(mask_raw,[1,0,2,3])
        return mask
    # '''
    
    def indicatorExpanding(points, stride_num) :
        # 把得到的缩小后Indicator（batch, width, height, channel）二维放大stride_num，输出（batch, width*stride_num, height*stride_num, channel）
        mask_h=points.get_shape()[1]
        mask_w=points.get_shape()[2]
        points_tr=tf.transpose(points,[2,0,1,3])
        mask_col=tf.gather(points_tr,sorted(stride_num*[i for i in range(mask_w)])[0:w])
        mask_tr=tf.transpose(mask_col,[2,1,0,3])
        mask_raw=tf.gather(mask_tr,sorted(stride_num*[i for i in range(mask_h)])[0:h])
        mask=tf.transpose(mask_raw,[1,0,2,3])
        return mask
    # '''

    mask = expand2D(points, stride_num)
    mask_bar=tf.subtract(mask_one, mask)
    
    countH_1 = tf.reduce_sum(mask, axis = [1,2,3])        # (None, )
    countA_1 = tf.reduce_sum(mask_one, axis = [1,2,3])
    # m_max = tf.reduce_max(points)
    # m_min = tf.reduce_min(points)
    percentage=countH_1 / tf.cast(w*h*c,'float32')*100    # (None, )
    
    input_h_mask=tf.multiply(input_h,mask)
    input_l_mask=tf.multiply(input_l,mask_bar)
    input_combine=tf.add(input_h_mask,input_l_mask)

    # 为了与keras兼容,统计量必须对每个输入进行,即统计输出为(None, 7), 百分比为(None, )
    
    imin_h = tf.reduce_min(input_h_mask, axis = [1,2,3])     # (None, )
    imax_h = tf.reduce_max(input_h_mask, axis = [1,2,3])     # (None, )
    imin_l = tf.reduce_min(input_l_mask, axis = [1,2,3])     # (None, )
    imax_l = tf.reduce_max(input_l_mask, axis = [1,2,3])     # (None, )
    fmin = tf.tile([tf.reduce_min(filter)], [s])      # (None, )
    fmax = tf.tile([tf.reduce_max(filter)], [s])      # (None, )


    fmin_run = tf.reduce_min(filter, axis = [0,1,2])      # (None, )
    fmax_run = tf.reduce_max(filter, axis = [0,1,2])     # (None, )
    
    fdiff=fmax_run-fmin_run

    p_h = tf.cast(fdiff >= filterGate, tf.float32)
    p_l = tf.cast(fdiff < filterGate, tf.float32)
    
    countH=tf.tile(  [  tf.reduce_sum(p_h)*tf.to_float(tf.size(filter[:,:,:,0]))],[s])
    countA=tf.tile( [tf.to_float(tf.size(filter))] , [s])

    filter_l_h = tf.multiply(filter_h[:,:,:,0:channels], p_h[0:channels])
    filter_l_l = tf.multiply(filter_l[:,:,:,0:channels], p_l[0:channels])

    #filter_l = filter_l_h + filter_l_l
    
    if quant:
        with tf.variable_scope('conv_sep'):
            convlayer_h = tf.nn.conv2d(
                input_h_mask, 
                filter_h, 
                ([1,] + list(strides) + [1,]), 
                padding, 
                dilations = ([1,] + list(dilations) + [1,]), 
            )
            convlayer_l = tf.nn.conv2d(
                input_l_mask, 
                filter_l, 
                ([1,] + list(strides) + [1,]), 
                padding, 
                dilations = ([1,] + list(dilations) + [1,]), 
            )
        convResult=tf.add(convlayer_h,convlayer_l)
    else:
        convResult=tf.nn.conv2d(
                input, 
                filter, 
                ([1,] + list(strides) + [1,]), 
                padding, 
                dilations = ([1,] + list(dilations) + [1,]), 
            )
        # percentage=tf.constant(0,dtype="float32")
        countH = tf.reduce_sum(mask, axis = [1,2,3])*0
        countA = tf.reduce_sum(mask, axis = [1,2,3])*1
    statics=tf.transpose(tf.convert_to_tensor([imin_h,imax_h,imin_l,imax_l,fmin,fmax,countH,countA,countH_1,countA_1]))      # 此处要与keras层的output_shape = (None, ５)相匹配
    
    return (convResult,percentage,statics)

'''
以下将量化卷积操作包装成Keras层
建立层时的不可变参数:
    - 输出行为
        - outputStaticInfo = False
    - 卷积行为: 
        - filters (cnt Output Channels)
        - Kernel Size
        - strides
        - padding
        - data_format == 'channels_last'
        - dilations
        - activation
        - use_bias
        - kernel_initializer='glorot_uniform',
        - bias_initializer='zeros',
        - kernel_regularizer=None,
        - bias_regularizer=None,
        - activity_regularizer=None,
        - kernel_constraint=None,
        - bias_constraint=None
    - 量化行为：
        - quant 是否量化
        - cntQuantPhase量化段数(1,2,3?)
        - gate = [0.25]
        - quant_input_minmax = [0,0]
        - quant_filter_minmax = [0,0]
        - num_bits = [8,4] 各段量化比特数
        - useQuantMinMaxVars = True
要建立的可训练参数：
    - 卷积核
    - 偏置
同时,建立并维护一个保存量化信息的对象quantRecord
'''
import tensorflow as tf
import keras
from keras.models import Model
import keras.backend as K
from keras.legacy import interfaces
from keras.layers.convolutional import _Conv    # 使用卷积层的基类

class classQuantControl() : 
    # 量化层的统计信息输出列表
    listPercentageOutput = []
    listStaticInfoOutput = []
    # 量化层参数列表
    listQuant = []                  # [bool, ...] 是否使用量化
    listCntQuantPhase = []          # [int  ...]
    listGate = []                   # [[float  ...]  ...] gate列表
    listfilterGate =[]
    listQuant_input_minmax = []     # [[int, int, ...], ...] Input_MinMax
    listQuant_filter_minmax = []    # [[int, int], ...]
    listNum_bits = []               # [[int, ...], ...]
    # 使用参数的情况
    pParam = 0
    
    def __init__(self) :
        return
    
    
    '''
    设置量化参数列表,并且重置指针
    '''
    def setQuantParamList(
                            self,
                            listQuant = [],                 # [bool, ...] 是否使用量化
                            listCntQuantPhase = [],         # [int, ...]
                            listGate = [], 
                            listfilterGate = [],                 # [[float, ...], ...] gate列表
                            listQuant_input_minmax = [],    # [[int, int, ...], ...] Input_MinMax
                            listQuant_filter_minmax = [],   # [[int, int], ...]
                            listNum_bits = [],              # [[int, ...], ...]
                        ) :
        # 要求提供的量化参数列表等长(表示相同数量的量化层)
        assert (len(listQuant) == len(listCntQuantPhase) == len(listGate) == len(listQuant_input_minmax) == len(listQuant_filter_minmax) == len(listNum_bits)), 'Given lists of quant parameter must have the same length'
        self.listQuant = listQuant
        self.listCntQuantPhase = listCntQuantPhase
        self.listGate = listGate
        self.listfilterGate = listfilterGate
        self.listQuant_input_minmax = listQuant_input_minmax
        self.listQuant_filter_minmax = listQuant_filter_minmax
        self.listNum_bits = listNum_bits
        print('Set', len(listQuant), 'layers'' param in quantControl.')
        self.pParam = 0
    
    '''
    重置指针,以复用参数列表
    '''
    def resetParamPointer(self) :
        self.pParam = 0
    
    '''
    从量化参数列表取得下一组参数,并且更新指针
    '''
    def getQuantParam(self) :
        if (self.pParam == len(self.listQuant)) :
            # All param in list have been used, or empty list
            print('WARNING # quant_util_keras.classQuantControl: getQuantParam found parameter list exhausted. Return defult param with NO QUANTILIZATION (quant = False).')
            return dict(
                            quant = False,
                            cntQuantPhase = 2,
                            gate = [0.25],
                            filterGate = 0.15,
                            quant_input_minmax = [0,0],
                            quant_filter_minmax = [0,0],
                            num_bits = [8,4],
                        )
        else :
            #print('quant_util_keras.classQuantControl: Using',self.pParam+1,'/',len(self.listQuant),'parameters in quantControl.')
            param = dict(
                            quant = self.listQuant[self.pParam],
                            cntQuantPhase = self.listCntQuantPhase[self.pParam],
                            gate = self.listGate[self.pParam],
                            filterGate = self.listfilterGate[self.pParam],
                            quant_input_minmax = self.listQuant_input_minmax[self.pParam],
                            quant_filter_minmax = self.listQuant_filter_minmax[self.pParam],
                            num_bits = self.listNum_bits[self.pParam],
                        )
            self.pParam = self.pParam + 1
            return param
    
    '''
    维护量化层统计信息列表,被keras层在build调用
    '''
    def addConvQuantInfo(self, percentage, statics) :
        self.listPercentageOutput.append(percentage)
        self.listStaticInfoOutput.append(statics)
    
    '''
    读取量化层统计信息列表,获得所有层的监测输出
    '''
    def getConvQuantInfo(self) :
        return (self.listPercentageOutput, self.listStaticInfoOutput)
    
    def clearConvQuantInfo(self) :
        self.listPercentageOutput = []
        self.listStaticInfoOutput = []

'''
建立量化控制对象
'''
quantControl = classQuantControl()

class Conv2DWithQuant(_Conv) :
    
    def setSelfQuantParam(  self,
                            quant = False,
                            cntQuantPhase = 2,                  # TODO: 目前仅支持2
                            gate = [0.25],                      # gate 应该按从高到低的顺序排列，对应精度从大到小
                            filterGate = 0.15,
                            # TODO: quant_input_minmax没有实现对多层的支持，max_l = max_h / 2
                            quant_input_minmax = [0,0],         # [min1, max1, min2, max2, ...]
                            quant_filter_minmax = [0,0],        # [min, max]
                            num_bits = [8,4],                   # num_bits应该按精度从高到低的顺序排列
                            useQuantMinMaxVars = True,
                        ) :
        self.quant = quant
        self.cntQuantPhase = cntQuantPhase              # TODO: 目前仅支持2
        self.gate = gate                                # gate 应该按从高到低的顺序排列，对应精度从大到小
        self.filterGate = filterGate
        # TODO: quant_input_minmax没有实现对多层的支持，max_l = max_h / 2
        self.quant_input_minmax = quant_input_minmax    # [min1, max1, min2, max2, ...]
        self.quant_filter_minmax = quant_filter_minmax  # [min, max]
        self.num_bits = num_bits                        # num_bits应该按精度从高到低的顺序排列
        self.useQuantMinMaxVars = useQuantMinMaxVars
    
    '''
    在初始化函数中要记录给定的初始化参数
    '''
    # @interfaces.legacy_conv2d_support
    def __init__(   self,
                    #>>>>>> Conv Parameter
                    filters,
                    kernel_size,
                    strides=(1, 1),
                    padding='valid',
                    data_format=None,
                    dilation_rate=(1, 1),
                    activation=None,
                    use_bias=True,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None,
                    #<<<<<< Conv Parameter
                    #>>>>>> Quant Parameter
                    useQuantControlParam = True,       # 使用quantControl中的量化设置,将会覆盖其他(以下6个)参数
                    quant = False,
                    cntQuantPhase = 2,                  # TODO: 目前仅支持2
                    gate = [0.25],                      # gate 应该按从高到低的顺序排列，对应精度从大到小
                    filterGate = 0.15,
                    # TODO: quant_input_minmax没有实现对多层的支持，max_l = max_h / 2
                    quant_input_minmax = [0,0],         # [min1, max1, min2, max2, ...]
                    quant_filter_minmax = [0,0],        # [min, max]
                    num_bits = [8,4],                   # num_bits应该按精度从高到低的顺序排列
                    useQuantMinMaxVars = True,
                    #<<<<<< Quant Parameter
                    #>>>>>> Behavior Parameter
                    outputStaticInfo = False,           # 将统计结果直接作为输出,层变为多输出的层
                    printDebug = True,
                    #<<<<<< Behavior Parameter
                    **kwargs
                ) :
        # 记录行为参数
        self.outputStaticInfo = outputStaticInfo
        self.printDebug = printDebug
        # 记录量化参数
        if useQuantControlParam :
            # 从quantControl读取参数
            self.setSelfQuantParam(** (quantControl.getQuantParam()), useQuantMinMaxVars = True)
        else :
            self.setSelfQuantParam(quant, cntQuantPhase, gate, filterGate, quant_input_minmax, quant_filter_minmax, num_bits, useQuantMinMaxVars)
        '''
        self.quant = quant
        self.cntQuantPhase = cntQuantPhase              # TODO: 目前仅支持2
        self.gate = gate                                # gate 应该按从高到低的顺序排列，对应精度从大到小
        # TODO: quant_input_minmax没有实现对多层的支持，max_l = max_h / 2
        self.quant_input_minmax = quant_input_minmax    # [min1, max1, min2, max2, ...]
        self.quant_filter_minmax = quant_filter_minmax  # [min, max]
        self.num_bits = num_bits                        # num_bits应该按精度从高到低的顺序排列
        self.useQuantMinMaxVars = useQuantMinMaxVars
        '''
        
        '''
        调用_Conv初始函数
        其行为主要是把参数进行适当的处理，保存到对应属性中
        '''
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
    
    def printd(self, *args) :
        if not self.printDebug :
            return
        print(*args)
    
    '''
    在build阶段要调用self.add_weight添加可训练（或不可训练）的变量
    同时要调用父类的build
        Layer基类的build只是设置了built标志
        _Conv的build
            - 没有调用父类的build方法，而是自行设置built标志
            - self.add_weight 添加了卷积核和偏置变量
    '''
    def build(self, input_shape):
        #self.printd('-----------------------------------------')
        #self.printd('Conv2DWithQuant build called')
        #self.printd('input_shape =',input_shape)
        #self.printd('-----------------------------------------')
        super().build(input_shape)
    
    '''
    在call阶段具体构造并返回tf计算图
    不能使用super().call
    '''
    def call(self, inputs):
        #self.printd('-----------------------------------------')
        #self.printd('Conv2DWithQuant call called')
        #self.printd('inputs =',inputs)
        #self.printd('-----------------------------------------')
        # super().call(x)
        '''
        outputs = K.conv2d(
            inputs,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)
        '''
        
        convResult,percentage,statics = buildConv2DLayer(
                inputs,
                channels=self.filters,
                filter = self.kernel,                           # 具体的卷积核Variable
                strides = self.strides,
                padding = self.padding.upper(),                   # tf中padding参数与keras不同，要大写
                dilations = self.dilation_rate,
                quant=self.quant,
                cntQuantPhase = self.cntQuantPhase,             # TODO: 目前仅支持2
                gate = self.gate,                    # gate 应该按从高到低的顺序排列，对应精度从大到小
                filterGate = self.filterGate,
                # TODO: quant_input_minmax没有实现对多层的支持，max_l = max_h / 2
                quant_input_minmax = self.quant_input_minmax,     # [min1, max1, min2, max2, ...]
                quant_filter_minmax = self.quant_filter_minmax,    # [min, max]
                num_bits = self.num_bits,               # num_bits应该按精度从高到低的顺序排列
                useQuantMinMaxVars = self.useQuantMinMaxVars,
                printDebug = self.printDebug,
            )
        
        outputs = convResult
        
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
        
        if self.activation is not None:
            outputs = self.activation(outputs)
        
        quantControl.addConvQuantInfo(percentage, statics)
        
        return (outputs if not self.outputStaticInfo else [outputs, percentage, statics])
    
    def compute_output_shape(self, input_shape):
        #self.printd('-----------------------------------------')
        #self.printd('Conv2DWithQuant compute_output_shape called')
        #self.printd('input_shape =',input_shape)
        shape_conv_output = super().compute_output_shape(input_shape)
        #self.printd('shape_conv_output =',shape_conv_output)
        #self.printd('-----------------------------------------')
        # 按照keras的约定,对batch中每个输入都要有输出,因此统计量要在输入上分别统计,而不能归为一个值
        return (shape_conv_output if not self.outputStaticInfo else [shape_conv_output, (None, ), (None, 10)])
    
    def get_config(self):
        #self.printd('-----------------------------------------')
        #self.printd('Conv2DWithQuant get_config called')
        #self.printd('-----------------------------------------')
        config = super().get_config()
        config.pop('rank')
        quant_config = dict(
            quant	            = self.quant,
            cntQuantPhase	    = self.cntQuantPhase,              
            gate	            = self.gate,
            filterGate          = self.filterGate,
            quant_input_minmax	= self.quant_input_minmax,     # [min1, max1, min2, max2, ...]
            quant_filter_minmax	= self.quant_filter_minmax,    # [min, max]
            num_bits	        = self.num_bits,               # num_bits应该按精度从高到低的顺序排列
            useQuantMinMaxVars	= self.useQuantMinMaxVars,
            
            outputStaticInfo	= self.outputStaticInfo,
        )
        config = dict(list(config.items()) + list(quant_config.items()))
        return config

'''
# input_t = tf.placeholder('float32',shape=(None,32,32,3))
input_t = keras.layers.Input((32,32,3))
output_t = Conv2DWithQuant(16,3, name = 'Foo_0_Bar_1')(input_t)

model = Model(inputs = [input_t], outputs = [output_t])
model.summary()
'''

'''
tLayer = Conv2DWithQuant(16,3, useQuantControlParam = True, outputStaticInfo = True)
'''

'''
能够自动消耗量化参数列表的量化卷积层
输入的量化参数为列表
    - quant_list = [bool]
    - cntQuantPhase_list = [int]
    - gate_list = [float]
    
'''
