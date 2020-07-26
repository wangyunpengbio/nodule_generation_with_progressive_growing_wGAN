import os
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dropout
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
import keras.backend as K
from keras.utils import plot_model
from keras.optimizers import RMSprop, SGD, Adam

from PIL import Image
import numpy as np
import glob
from random import randint, shuffle
import random

import time
from IPython.display import clear_output
from IPython.display import display


# Weights initializations
# bias are initailized as 0
def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a) # for convolution kernel
    k.conv_weight = True    
    return k

# Basic discriminator
def conv2d(f, *a, **k):
    return Conv2D(f, kernel_initializer = conv_init, *a, **k)
# channel_axis=-1
def batchnorm():
    return BatchNormalization(momentum=0.9, axis=channel_axis, epsilon=1.01e-5,
                                   gamma_initializer = gamma_init)
def BASIC_D(nc_in, nc_out, ndf, max_layers=3):
    """DCGAN_D(nc, ndf, max_layers=3)
       nc: channels
       ndf: filters of the first layer
       max_layers: max hidden layers
    """ 
    # channel_first为False
    if channel_first:
        input_a, input_b =  Input(shape=(nc_in, None, None)), Input(shape=(nc_out, None, None))
    else:
        input_a, input_b = Input(shape=(None, None, nc_in)), Input(shape=(None, None, nc_out))
    #print(input_a,input_b)
    _ = Concatenate(axis=channel_axis)([input_a, input_b])
    _ = conv2d(ndf, kernel_size=4, strides=2, padding="same", name = 'First') (_)
    _ = LeakyReLU(alpha=0.2)(_)
    
    for layer in range(1, max_layers):        
        out_feat = ndf * min(2**layer, 8)
        _ = conv2d(out_feat, kernel_size=4, strides=2, padding="same", 
                   use_bias=False, name = 'pyramid.{0}'.format(layer)             
                        ) (_)
        _ = batchnorm()(_, training=1)        
        _ = LeakyReLU(alpha=0.2)(_)
    
    out_feat = ndf*min(2**max_layers, 8)
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(out_feat, kernel_size=4,  use_bias=False, name = 'pyramid_last') (_)
    _ = batchnorm()(_, training=1)
    _ = LeakyReLU(alpha=0.2)(_)
    # final layer
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(1, kernel_size=4, name = 'final'.format(out_feat, 1), 
               activation = "sigmoid") (_)
    return Model(inputs=[input_a, input_b], outputs=_)

def UNET_G(isize, nc_in=3, nc_out=3, ngf=64, fixed_input_size=True):    
    max_nf = 8*ngf    
    def block(x, s, nf_in, use_batchnorm=True, nf_out=None, nf_next=None):
        assert s>=2 and s%2==0
        if nf_next is None:
            nf_next = min(nf_in*2, max_nf)
        if nf_out is None:
            nf_out = nf_in
        x = conv2d(nf_next, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s>2)),
                   padding="same", name = 'conv_{0}'.format(s)) (x)
        if s>2:
            if use_batchnorm:
                x = batchnorm()(x, training=1)
            x2 = LeakyReLU(alpha=0.2)(x)
            x2 = block(x2, s//2, nf_next)
            x = Concatenate(axis=channel_axis)([x, x2])            
        x = Activation("relu")(x)
        x = Conv2DTranspose(nf_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                            kernel_initializer = conv_init,          
                            name = 'convt.{0}'.format(s))(x)
        # 剪裁
        # cropping：长为2的整数tuple，分别为宽和高方向上头部与尾部需要裁剪掉的元素数
        x = Cropping2D(1)(x)
        if use_batchnorm:
            x = batchnorm()(x, training=1)
        if s <=8:
            x = Dropout(0.5)(x, training=1)
        return x
    
    # s就是isize，即图像大小，fixed_input_size为True
    s = isize if fixed_input_size else None
    if channel_first:
        _ = inputs = Input(shape=(nc_in, s, s))
    else:
        _ = inputs = Input(shape=(s, s, nc_in))
    _ = block(_, isize, nc_in, False, nf_out=nc_out, nf_next=ngf)
    _ = Activation('tanh')(_)
    return Model(inputs=inputs, outputs=[_])


def load_data(file_pattern):
    return glob.glob(file_pattern)

def read_image(fn, direction=0):
    im = np.load(fn)
    # 归一化到[-1,1]
    arr = np.array(im)/255*2-1#输入是0-255，归一化到-1到1
    # imgA为arr[15:271,286+15,286+271],正好shape为256*256
    imgA = arr[0:imageSize, imageSize:2*imageSize, np.newaxis]
    # imgB为arr[15:271,15:271],正好shape为256*256
    imgB = arr[0:imageSize, 0:imageSize, np.newaxis]
    if randint(0,1):
        # 是否需要左右对换
        imgA=imgA[:,::-1]
        imgB=imgB[:,::-1]
    if channel_first:
        imgA = np.moveaxis(imgA, 2, 0)
        imgB = np.moveaxis(imgB, 2, 0)
    if direction==0:
        return imgA, imgB
    else:
        return imgB, imgA

def netG_gen(A):
    return np.concatenate([netG_generate([A[i:i+1]])[0] for i in range(A.shape[0])], axis=0)

def minibatch(dataAB, batchsize, direction=0):
    length = len(dataAB)
    epoch = i = 0
    tmpsize = None    
    while True:
        size = tmpsize if tmpsize else batchsize
        if i+size > length:
            shuffle(dataAB)
            i = 0
            epoch+=1        
        dataA = []
        dataB = []
        for j in range(i,i+size):
            imgA,imgB = read_image(dataAB[j], direction)
            dataA.append(imgA)
            dataB.append(imgB)
        dataA = np.float32(dataA)
        dataB = np.float32(dataB)
        i+=size
        # send(msg) 和 next()是有返回值的，它们的返回值很特殊，返回的是下一个yield表达式的参数。
        # send()的两个功能：1.传值；2.next()。t.__next__()相当于send(None)
        # tmpsize为send()函数填入的值，即为6
        tmpsize = yield epoch, dataA, dataB
        # 但是需要注意，在一个生成器对象没有执行next方法之前，由于没有yield语句被挂起，所以执行send方法会报错。
        # 所以，第一次运行只能使用next或者send(None)


# 此函数修改成，只能处理灰度图（原来函数处理灰度图，会产生RGB颜色）
def showX(X, rows=1):
    assert X.shape[0]%rows == 0
    int_X = ( (X+1)/2*255 ).clip(0,255).astype('uint8')
    int_X = np.hstack(int_X).squeeze()
    #display(Image.fromarray(int_X))
    return Image.fromarray(int_X).convert('L')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND']='tensorflow' # can choose theano, tensorflow, cntk
os.environ['THEANO_FLAGS']='floatX=float32,device=cuda,optimizer=fast_run,dnn.library_path=/usr/lib'


workdir = "/data/wangyunpeng/3class/pix2pix"
noduleclass = 'AIS'
if not os.path.exists(os.path.join(workdir,noduleclass,"weight")):
    os.makedirs(os.path.join(workdir,noduleclass,"weight"))
if not os.path.exists(os.path.join(workdir,noduleclass,"png")):
    os.makedirs(os.path.join(workdir,noduleclass,"png"))
if not os.path.exists(os.path.join(workdir,noduleclass,"log")):
    os.makedirs(os.path.join(workdir,noduleclass,"log"))

if os.environ['KERAS_BACKEND'] =='theano':
    channel_axis=1
    K.set_image_data_format('channels_first')
    channel_first = True
else:
    K.set_image_data_format('channels_last')
    channel_axis=-1
    channel_first = False


# create a tf session，and register with keras。
sess = tf.Session()
K.set_session(sess)


conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02) # for batch normalization


# HACK speed up theano
if K._BACKEND == 'theano':
    import keras.backend.theano_backend as theano_backend
    def _preprocess_conv2d_kernel(kernel, data_format):
        #return kernel
        if hasattr(kernel, "original"):
            print("use original")
            return kernel.original
        elif hasattr(kernel, '_keras_shape'):
            s = kernel._keras_shape
            print("use reshape",s)
            kernel = kernel.reshape((s[3], s[2],s[0], s[1]))
        else:
            kernel = kernel.dimshuffle((3, 2, 0, 1))
        return kernel
    theano_backend._preprocess_conv2d_kernel = _preprocess_conv2d_kernel

nc_in = 1
nc_out = 1
ngf = 128
ndf = 128
λ = 10

# loadSize = 286
imageSize = 64
batchSize = 32
lrD = 1e-4
lrG = 1e-4


netD = BASIC_D(nc_in, nc_out, ndf)
# netD.summary()


netG = UNET_G(imageSize, nc_in, nc_out, ngf)
# netG.summary()
plot_model(netG, to_file=os.path.join(workdir,noduleclass,"log",'pix2pix_generator.png'))
plot_model(netD, to_file=os.path.join(workdir,noduleclass,"log",'pix2pix_discriminator.png'))



real_A = netG.input
fake_B = netG.output
netG_generate = K.function([real_A], [fake_B])
real_B = netD.inputs[1]
output_D_real = netD([real_A, real_B])
output_D_fake = netD([real_A, fake_B])


loss_fn = lambda output, target : -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target))

loss_D_real = loss_fn(output_D_real, K.ones_like(output_D_real)*random.uniform(0.8, 1.1))
loss_D_fake = loss_fn(output_D_fake, K.zeros_like(output_D_fake)*random.uniform(0, 0.2))
loss_G_fake = loss_fn(output_D_fake, K.ones_like(output_D_fake))

loss_L1 = K.mean(K.square(fake_B-real_B))

loss_D = loss_D_real + loss_D_fake
training_updates_D = Adam(lr=lrD, beta_1=0.5).get_updates(netD.trainable_weights,[],loss_D)


loss_G = loss_G_fake   + 1000 * loss_L1
training_updates_G = Adam(lr=lrG, beta_1=0.5).get_updates(netG.trainable_weights,[], loss_G)


direction = 0
trainAB = load_data(os.path.join('/data/wangyunpeng/3class','trash-2/pix2pix/datasets/train',noduleclass,'*.npy'))#此处输入是0-255，归一化到-1到1
valAB = load_data(os.path.join('/data/wangyunpeng/3class','trash-2/pix2pix/datasets/val',noduleclass,'*.npy'))
assert len(trainAB) and len(valAB)


# 试着展示一波样本
train_batch = minibatch(trainAB, 6, direction=direction)
_, trainA, trainB = next(train_batch)
showX(trainA)
showX(trainB)
del train_batch, trainA, trainB


# tensorboard
writer = tf.summary.FileWriter(os.path.join(workdir,noduleclass,'log'),sess.graph)
# discriminator需要记录的参数(输入图像就不记录了)
# netD_real_input_shaped = tf.summary.image('real_img', netD_real_input_shaped, 3)
loss_D_real = tf.summary.scalar('loss_D_real', loss_D_real)
loss_D_fake = tf.summary.scalar('loss_D_fake', loss_D_fake)
loss_D = tf.summary.scalar('loss_D', loss_D)
merged_dis = tf.summary.merge([loss_D_real,loss_D_fake,loss_D])#netD_real_input_shaped,
# generator需要记录的参数()
# 最后的那个参数表示记录的图像数目
real_A_record = tf.summary.image('real_A', real_A, 1)
fake_B = tf.summary.image('fake_img', fake_B, 1)
loss_L1 = tf.summary.scalar('loss_L1', loss_L1)
loss_G_fake = tf.summary.scalar('loss_G_fake', loss_G_fake)
loss_G = tf.summary.scalar('loss_G', loss_G)
merged_gen = tf.summary.merge([loss_L1,loss_G_fake,loss_G,real_A_record,fake_B])


# 必须要放在定义完所有图的后面，然后在加载以前训练过的权值的前面
# 初始化session中的变量
sess.run(tf.global_variables_initializer())

t0 = time.time()
niter = 500
gen_iterations = 0
errL1 = epoch = errG = 0
errL1_sum = errG_sum = errD_sum = 0
#用于计数写入文件的次数
writer_global_step = 0

display_iters = 5
val_batch = minibatch(valAB, 6, direction)
train_batch = minibatch(trainAB, batchSize, direction)

save_epoch = 0
while epoch < niter: 
    epoch, trainA, trainB = next(train_batch)
    #errG, errL1 = netG_train([trainA, trainB])
    #netG_train = K.function([real_A, real_B], [loss_G_fake, loss_L1], training_updates)
    summary, _ = sess.run([merged_gen, training_updates_G],feed_dict={real_A:trainA, real_B:trainB})
    writer.add_summary(summary, writer_global_step)
    writer.flush()
    
    #errD,  = netD_train([trainA, trainB])
    #netD_train = K.function([real_A, real_B],[loss_D/2], training_updates)
    summary, _ = sess.run([merged_dis, training_updates_D],feed_dict={real_A:trainA, real_B:trainB})
    writer.add_summary(summary, writer_global_step)
    writer_global_step+=1
    writer.flush()
    gen_iterations+=1

    # 每display_iters次，展示一波，每展示10波，清空控制台输出
    if gen_iterations%display_iters==0:
        if gen_iterations%(10*display_iters)==0:
            clear_output()
        print('[%d/%d][%d]' % (epoch, niter, gen_iterations), time.time()-t0)
        if epoch > save_epoch:
            netG.save(os.path.join(workdir,noduleclass,'weight','generator_weights_%05d.h5' % epoch))
            netD.save(os.path.join(workdir,noduleclass,'weight','discriminator_weights_%05d.h5' % epoch))
            save_epoch = epoch
        # 训练集数据可视化
        _, valA, valB = train_batch.send(6) 
        fakeB = netG_gen(valA)
        trainimg = showX(np.concatenate([valA, valB, fakeB], axis=0), 3)
        trainimg.save(os.path.join(workdir,noduleclass,'png','train_%05d.png' % epoch))
        # 交叉验证集数据可视化
        _, valA, valB = next(val_batch)
        fakeB = netG_gen(valA)
        valimg = showX(np.concatenate([valA, valB, fakeB], axis=0), 3)
        valimg.save(os.path.join(workdir,noduleclass,'png','val_%05d.png' % epoch))
        