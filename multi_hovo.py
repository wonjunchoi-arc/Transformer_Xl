

##data load ### cpu 버전

import os
import pickle
import tensorflow as tf
import glob
import numpy as np
import matplotlib.pyplot as plt
import time
from data_cpu import Dataset, LMOrderedIterator
from model import TFTransfoXLModel,TFTransfoXLLMHeadModel
from tensorflow.keras.utils import register_keras_serializable

from transformers import TransfoXLConfig



import horovod.tensorflow as hvd

# Horovod 초기화
hvd.init()

# GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


config_xl = TransfoXLConfig(
    data = '/home/jun/workspace/wiki_short/',
    dataset = 'wt103',
    d_embed=128,
    d_head = 32,
    d_model=128,
    mem_len=400,
    n_head=8,
    n_layer=6,
    batch_size = 24,
    tgt_len = 60,
    ext_len = 0,
    eval_tgt_len = 60
)



kwargs = {}
if config_xl.dataset in ['wt103', 'wt2']:
    kwargs['special'] = ['<eos>','<UNK>']
    kwargs['lower_case'] = False

dataset = Dataset(**kwargs)

train_dataset, val_dataset, test_dataset = dataset.make_dataset(config_xl.data,config_xl.dataset)

# strategy = tf.distribute.MirroredStrategy()

def gen(data,bsz,bptt,ext_len=None,):
  
  bsz = bsz#3 #60
  bptt = bptt#36 #70
  ext_len = ext_len if ext_len is not None else 0
  data = data
  
  
  # Work out how cleanly we can divide the dataset into bsz parts.
  # 아래의 두 코드는   data 텐서에서 배치 크기 bsz로 깔끔하게 맞지 않는 추가 요소를 제거하는 것 배치에 띡 떨어지게
  n_step = len(data) // (bsz*bptt)
  print('n_step',n_step) # 312779
  
  sliced_data = tf.slice(data,[0],[n_step * bsz*bptt])  
  # print('sliced_data',len(sliced_data))
  # sliced_data = self.data[:self.n_step * self.bsz]
  '''# 시작 위치와 슬라이싱할 크기 설정
  begin = [0]  # 첫 번째 차원의 시작 위치는 0
  size = [6]   # 첫 번째 차원에서 6개의 원소를 슬라이싱

  # 데이터를 잘라내기 (tf.slice 사용)
  sliced_data = tf.slice(data, begin, size)  '''

  # Evenly divide the da
  # ta across the bsz batches.


  new_shape = (bsz, -1)  # 나머지 차원은 자동으로 계산됨
  data_reshaped = tf.reshape(sliced_data, new_shape)
  # data_transposed = tf.transpose(data_reshaped)
  data = data_reshaped
  # print('data',len(data))
  split_num = 2 #GPU num


  # first_half, second_half = tf.split(data, num_or_size_splits=split_num, axis=1)

  n_batch = (n_step + bptt - 1) // bptt

  for i in range(0, len(data[1]) - 1, bptt):
    
    seq_len = min(bptt, data.shape[1] - 1 - i) # # i값이 103227020를 넘지 않는 이상 seq_len = 70


    end_idx = i + seq_len # 70,71,72,73,74......
    beg_idx = max(0, i - ext_len) # 0,1,2,3,4,5
    ''' 아래 처럼 첫번째 차원을 자르는 이류
    로,또,1,등,당,첨 = > 로,또,1    => 로, 등
                    등,당,첨         또, 당
                                    1, 첨
    '''

    p_data = data[:,beg_idx:end_idx] # self.data[:,0:70],[:,1:71] ~
    target = data[:,i+1:i+1+seq_len]

    # second_half_data = second_half[:,beg_idx:end_idx] # self.data[:,0:70],[:,1:71] ~
    # second_half_target = second_half[:,i+1:i+1+seq_len]
    if i + bptt < len(data[1]) - 1:
      yield p_data, target
      # yield second_half_data, second_half_target


dataset = tf.data.Dataset.from_generator(
     gen,
     output_signature=(
         tf.TensorSpec(shape=None, dtype=tf.int32),
         tf.TensorSpec(shape=None, dtype=tf.int32),
         ),
     args=(train_dataset,config_xl.batch_size,config_xl.tgt_len)
         )


count =0 
for sample in dataset:
    count += 1
print(count)

first_half_dataset = dataset.take(count//2)
second_half_dataset = dataset.skip((count // 2) + (count % 2))


@register_keras_serializable()
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)

        self.warmup_steps = tf.cast(warmup_steps,tf.float32)

    def __call__(self, step):
        step =tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
    
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)*hvd.size()
    
    def get_config(self):
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
            }
learning_rate = CustomSchedule(config_xl.d_model)




# CustomSchedule 및 모델 정의는 이전과 동일하게 유지합니다.
# ...

# 옵티마이저 정의 및 Horovod 래핑
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
optimizer = hvd.DistributedOptimizer(optimizer)

# 모델 정의

# 모델 및 데이터셋 생성
if hvd.rank() == 0:
    model0 = TFTransfoXLLMHeadModel(config=config_xl)
  # GPU:0에서 사용할 첫 번째 모델
    dataset0 = first_half_dataset       # 첫 번째 데이터셋
    mems0 = None              # 첫 번째 모델의 메모리 상태
elif hvd.rank() == 1:
    model1 = TFTransfoXLLMHeadModel(config=config_xl)
  # GPU:1에서 사용할 두 번째 모델
    dataset1 = second_half_dataset       # 두 번째 데이터셋
    mems1 = None              # 두 번째 모델의 메모리 상태

# 변수 초기화 및 방송
# hvd.broadcast_variables(model0.variables, root_rank=0)
# hvd.broadcast_variables(optimizer1.variables(), root_rank=0)
# hvd.broadcast_variables(model1.variables, root_rank=0)
# hvd.broadcast_variables(optimizer2.variables(), root_rank=0)

# 훈련 스텝 정의
@tf.function
def train_step(model, data, target, mems, optimizer,first_batch):
    with tf.GradientTape() as tape:
        outputs = model(input_ids=data, labels=target, mems=mems)
        loss = outputs.loss[0]
        mems = outputs.mems
    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if first_batch:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(optimizer.variables(), root_rank=0)

    return mems,loss


    return mems, loss


for epoch in range(2):
    start = time.time()
    num_batches = 0
    total_loss = 0.0
    if hvd.rank() == 0:
        for data, target in dataset0:
            mems0, loss_value = train_step(model0, data, target, mems0, optimizer,num_batches==0)
            num_batches += 1
            total_loss += loss_value.numpy()

    elif hvd.rank() == 1:
        for data, target in dataset1:
            mems1, loss_value = train_step(model1, data, target, mems1, optimizer,num_batches==0)
            num_batches += 1
            total_loss += loss_value.numpy()
            if num_batches % 100 == 0:
                print(f'Epoch {epoch + 1} Batch {num_batches} Loss {loss_value.numpy()}')


# # 훈련 루프
# for epoch in range(2):
    
#     total_loss = 0.0
#     num_batches = 0

#     # 데이터셋을 반으로 나눠서 각 GPU에서 처리합니다.
#     for data,target in dataset:
#         mems1, loss = train_step(model0, data, target, mems1, optimizer1)

#     # for (first_data, first_target), (second_data, second_target) in zip(first_half_dataset, second_half_dataset):
#     #     if hvd.rank() == 0:
#     #         mems1, loss = train_step(model0, first_data, first_target, mems1, optimizer1)
#     #         total_loss += loss.numpy()

#     #     elif hvd.rank() == 1:
#     #         mems2, loss = train_step(model1, second_data, second_target, mems2, optimizer2)
#     #         total_loss += loss.numpy()

#         num_batches += 1

#         if num_batches % 100 == 0:
#             print(f'Epoch {epoch + 1} Batch {num_batches} Loss {loss.numpy()}')

#     # 에포크마다 평균 손실 계산
#     avg_loss = total_loss / num_batches
#     print(f'Epoch {epoch + 1}, Loss: {avg_loss}, Time: {time.time() - start}')
