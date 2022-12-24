# %% 导入包
import pretty_midi
import tensorflow as tf
import os
import random

seq_length = 24 # 输入序列长度
vocab_size = 128 # 分类数量
checkpoint_path = 'model/model.ckpt'  # 模型存放路径

# 制作数据集 ==================================================
def read_midi_notes():
  midi_inputs = [] # 存放所有的音符
  filenames = tf.io.gfile.glob("datasets/*.midi")
  # 循环所有midi文件
  for f in filenames:
    pm = pretty_midi.PrettyMIDI(f) # 加载一个文件
    instruments = pm.instruments # 获取乐器
    instrument = instruments[0] # 取第一个乐器，此处是原声大钢琴
    notes = instrument.notes # 获取乐器的演奏数据
    # 以开始时间start做个排序。因为默认是依照end排序
    sorted_notes = sorted(notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start
    # 循环各项指标，取出前后关联项
    for note in sorted_notes: 
      step =  note.start - prev_start # 此音符与上一个距离
      duration = note.end - note.start # 此音符的演奏时长
      prev_start = note.start # 此音符开始时间作为最新
      # 指标项：[音高（音符），同前者的间隔，自身演奏的间隔]
      midi_inputs.append([note.pitch, step, duration])

  return midi_inputs

# 将序列拆分为输入和输出标签对
def split_labels(sequences):
  inputs = sequences[:-1]
  inputs_x = inputs/[vocab_size,1.0,1.0]
  inputs_y = sequences[-1]
  labels = {"pitch":inputs_y[0], "step":inputs_y[1], "duration":inputs_y[2]}
  return inputs_x, labels

# 将数据组装为格式化的训练数据
def get_train_data(midi_inputs):
  # 搞成tensor，便于流操作，比如notes_ds.window
  notes_ds = tf.data.Dataset.from_tensor_slices(midi_inputs)
  cut_seq_length = seq_length+1 # 截取的长度，因为要拆分为输入+输出，因此+1
  # 每次滑动一个数据，每次截取cut_seq_length个长度
  windows = notes_ds.window(cut_seq_length, shift=1, stride=1,drop_remainder=True)
  flatten = lambda x: x.batch(cut_seq_length, drop_remainder=True)
  sequences = windows.flat_map(flatten)
  # 将25，拆分为24+1。24是输入，1是预测。进行训练
  seq_ds = sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)
  n_notes = len(midi_inputs)
  batch_size = 64
  buffer_size = n_notes - seq_length
  # 拆分批次，缓存等优化
  train_ds = (seq_ds.shuffle(buffer_size)
              .batch(batch_size, drop_remainder=True)
              .cache().prefetch(tf.data.experimental.AUTOTUNE))
  return train_ds

# 构建模型 ==================================================
# 自写损失函数
def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
  mse = (y_true - y_pred) ** 2
  positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse + positive_pressure)

# 创建模型
def create_model():

  input_shape = (seq_length, 3) # 输入形状
  inputs = tf.keras.Input(input_shape)
  x = tf.keras.layers.LSTM(128)(inputs)
  outputs = {
    'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
    'step': tf.keras.layers.Dense(1, name='step')(x),
    'duration': tf.keras.layers.Dense(1, name='duration')(x),
  }
  model = tf.keras.Model(inputs, outputs)
  loss = {
    'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    'step': mse_with_positive_pressure,
    'duration': mse_with_positive_pressure,
  }
  model.compile(
      loss=loss,
      loss_weights={'pitch': 0.05,'step': 1.0,'duration':1.0},
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
  )
  if os.path.exists(checkpoint_path + '.index'):
    print("---load model---")
    model.load_weights(checkpoint_path)  

  return model

# 训练样本 ==================================================
def train():
  midi_inputs = read_midi_notes()
  train_ds = get_train_data(midi_inputs)
  model = create_model()
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path
      ,save_weights_only=True, save_best_only=True)
  model.fit(train_ds, validation_data=train_ds, epochs=50,callbacks=[cp_callback])
  
#  预测数据 ================================================================
def predict_midi():
  model = create_model() # 加载模型
  # 读取样本
  midi_inputs = read_midi_notes()
  # 从音符库中随机拿出24个音符。当然你也可以自己编
  sample_notes = random.sample(midi_inputs, seq_length)
  num_predictions = 600 # 预测后面600个
  # 循环600次
  for i in range(num_predictions):
    notes = []
    # 拿出最后24个
    n_notes = sample_notes[-seq_length:]
    # 主要给音高做一个128分类
    for input in n_notes:
      notes.append([input[0]/vocab_size,input[1],input[2]])
    # 将24个音符交给模型预测
    predictions = model.predict([notes])
    # 取出预测结果
    pitch_logits = predictions['pitch']
    pitch = tf.random.categorical(pitch_logits, num_samples=1)[0]
    step = predictions['step'][0]
    duration = predictions['duration'][0]
    pitch, step, duration = int(pitch), float(step), float(duration)
    # 添加到音符数组中
    sample_notes.append([pitch, step, duration])

  print(sample_notes)

  # 复原midi数据
  prev_start = 0
  midi_notes = []
  for m in sample_notes:
    pitch, step, duration = m
    start = prev_start + step
    end = start + duration
    prev_start = start
    midi_notes.append([pitch, start, end])

  print(midi_notes)

  # 写入midi文件
  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano"))
  for n in midi_notes:
    note = pretty_midi.Note(velocity=100,pitch=n[0],start=n[1],end=n[2])
    instrument.notes.append(note)
  pm.instruments.append(instrument)
  pm.write("out.midi")
# %%
predict_midi()

# %%
