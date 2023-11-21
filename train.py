##data load ### cpu 버전

import os
import pickle
import tensorflow as tf
import glob
import numpy as np
import matplotlib.pyplot as plt
import time
from data import Dataset, LMOrderedIterator
from model import TFTransfoXLModel,TFTransfoXLLMHeadModel
import time
from transformers import TransfoXLConfig


config_xl = TransfoXLConfig(
    data = '/home/jun/workspace/wiki_short/',
    dataset = 'wt103',
    d_embed=128,
    d_head = 32,
    d_model=128,
    mem_len=400,
    n_head=8,
    n_layer=6,
    batch_size = 18,
    tgt_len = 36,
    ext_len = 0,
    eval_tgt_len = 70
)



kwargs = {}
if config_xl.dataset in ['wt103', 'wt2']:
    kwargs['special'] = ['<eos>','<UNK>']
    kwargs['lower_case'] = False

dataset = Dataset(**kwargs)

train_dataset, val_dataset, test_dataset = dataset.make_dataset(config_xl.data,config_xl.dataset)

# strategy = tf.distribute.MirroredStrategy()

# train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
# val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)
# test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)



ntokens = len(dataset.sym2idx.items())
n_token = ntokens
config_xl.n_token

# data_len = config.tgt_len * 20 # 20이 Segment의 갯수를 나타냅니다.
device='gpu:0'
eval_batch_size = 10
tr_iter = LMOrderedIterator(train_dataset, config_xl.batch_size, config_xl.tgt_len,
    device=device, ext_len=config_xl.ext_len)
va_iter = LMOrderedIterator(val_dataset, eval_batch_size, config_xl.eval_tgt_len,
    device=device, ext_len=config_xl.ext_len)
te_iter = LMOrderedIterator(test_dataset, eval_batch_size, config_xl.eval_tgt_len,
    device=device, ext_len=config_xl.ext_len)


# tr_iter = strategy.experimental_distribute_dataset(tr_iter)









class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = tf.cast(d_model, tf.float32)

    self.warmup_steps = tf.cast(warmup_steps,tf.float32)

  def __call__(self, step):
    step =tf.cast(step, tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  
learning_rate = CustomSchedule(config_xl.d_model)

# with strategy.scope():
model = TFTransfoXLLMHeadModel(config=config_xl)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                  epsilon=1e-9)


# with strategy.scope():

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')



checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=model,
                          optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!!')


def train_step(data, target,mems):
  

  with tf.GradientTape() as tape:
    outputs = model(input_ids=data,labels=target,mems=mems)
    loss = outputs.loss
    mems = outputs.mems

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  loss = train_loss(loss)
  return mems,loss

# @tf.function
# def distributed_train_step(dist_inputs):
#   per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
#   return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
#                          axis=None)



for epoch in range(6):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()

  # inp -> portuguese, tar -> english
  mems = None
  for batch, (data, target, seq_len) in enumerate(tr_iter):
    start_time = time.time()
    # data , target= tf.convert_to_tensor(data),tf.convert_to_tensor(target)
    mems = train_step(data, target,mems)
    end_time = time.time()

    if batch % 50 == 0:
      print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f}')

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

  print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

  print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

