import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # 指定CUDA可以看到的GPU，编号从0开始
# OS environment的代码要放在tensorflow init的上面。
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True) #设置CUDA根据模型大小占用内存
    except RuntimeError as e:
        print(e)
print(tf.__version__)

# CUDA占用内存的逻辑非常的捉急：
# 如果我看到有一个GPU，我会把这个GPU的所有空余内存都拿过来，不论模型的大小
# 如果我看到有多个GPU（如编号0，1，2，3），我会把编号最低的（0）所有空余内存都拿过来，不论模型大小
# 并且所有的运算都在编号最低的显卡上进行。
# 同时，我会占用其他GPU少量的内存，当编号最低显卡内存不够的话，我会依次拿走其他显卡的所有内存。

# 然而，除了某些极大模型之外，通常我们的模型不会占用超过24GB的内存，也不会让CUDA充分利用2个以上显卡的算力
# 那么我们能做的是首先限制CUDA只使用我们所需的几个显卡（有必要的情况下），并且让CUDA只使用模型需要的内存。

# 同时，应当尽可能避免两个tensorflow程序在一个显卡算力下运行。
