import os
import io
import time
import pickle
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf
import sentencepiece as spm
import tensorflow_text as tf_text
import matplotlib.pyplot as plt

path = os.path.dirname(__file__)

datasetIndex = 2

try:
    with open(f'{path}/dataset.pickle', 'rb') as f:
        datasetList = pickle.load(f)

    dataset = datasetList[datasetIndex]

    train_dataset = tf.data.Dataset.from_tensor_slices(dataset['train']).map(lambda d: (d[0], d[1]))
    test_dataset = tf.data.Dataset.from_tensor_slices(dataset['test']).map(lambda d: (d[0], d[1]))

    model = open(f'{path}/sentencepiece/tokenizer{datasetIndex}.model', 'rb').read()

    tokenizer = tf_text.SentencepieceTokenizer(model=model)
except:
    dataPairs = list()

    data = io.open(f'{path}/data/train.txt').read().split('\n')[1:-1]

    for sentence in data:
        p = sentence.split('|')
        dataPairs.append((' '.join(p[4:]), p[3]))

    np.random.shuffle(dataPairs)
    
    datasetList = list()

    testDataLength = len(dataPairs) // 4

    pathlib.Path(f'{path}/sentencepiece').mkdir(exist_ok=True)

    for i in range(4):
        trainData = dataPairs[:testDataLength*i] + dataPairs[testDataLength*(i+1):]
        testData = dataPairs[testDataLength*i:testDataLength*(i+1)]
    
        datasetList.append({'train': trainData, 'test': testData})

        with open(f'{path}/sentencepiece/sentence{i}.txt', 'w', encoding='utf-8') as f:
            for d in trainData:
                print(d[0], file=f)

        spm_train_args = dict(
            input=f'{path}/sentencepiece/sentence{i}.txt',
            model_prefix=f'{path}/sentencepiece/tokenizer{i}',
            vocab_size=4096,
            character_coverage=1.0
        )

        spm.SentencePieceTrainer.Train(**spm_train_args)
        
        if i == datasetIndex:
            train_dataset = tf.data.Dataset.from_tensor_slices(trainData).map(lambda d: (d[0], d[1]))
            test_dataset = tf.data.Dataset.from_tensor_slices(testData).map(lambda d: (d[0], d[1]))

            model = open(f'{path}/sentencepiece/tokenizer{i}.model', 'rb').read()

            tokenizer = tf_text.SentencepieceTokenizer(model=model)

    with open(f'{path}/dataset.pickle', 'wb') as f:
        pickle.dump(datasetList, f)

# TEMP
# dataPairs = list()

# data = io.open(f'{path}/data/train.txt').read().split('\n')[1:-1]

# for sentence in data:
#     p = sentence.split('|')
#     dataPairs.append((' '.join(p[4:]), p[3]))

# np.random.shuffle(dataPairs)

# # testDataLength = int(len(dataPairs) * 0.1)

# pathlib.Path(f'{path}/sentencepiece').mkdir(exist_ok=True)

# # trainData = dataPairs[testDataLength:]

# with open(f'{path}/sentencepiece/sentence_temp.txt', 'w', encoding='utf-8') as f:
#     for d in dataPairs:
#         print(d[0], file=f)

# spm_train_args = dict(
#     input=f'{path}/sentencepiece/sentence_temp.txt',
#     model_prefix=f'{path}/sentencepiece/tokenizer_temp',
#     vocab_size=4096,
#     character_coverage=1.0
# )

# spm.SentencePieceTrainer.Train(**spm_train_args)

# dataset = tf.data.Dataset.from_tensor_slices(dataPairs).map(lambda d: (d[0], d[1]))

# train_dataset = dataset
# test_dataset = dataset
# # test_dataset = dataset.take(testDataLength)
# # train_dataset = dataset.skip(testDataLength)

model = open(f'{path}/sentencepiece/tokenizer_temp.model', 'rb').read()

tokenizer = tf_text.SentencepieceTokenizer(model=model)
# TEMP

print(len(train_dataset), len(test_dataset))

START = tokenizer.string_to_id('<s>')
END = tokenizer.string_to_id('</s>')

def add_start_end(ragged):
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count, 1], START)
    ends = tf.fill([count, 1], END)
    return tf.concat([starts, ragged, ends], axis=1)

def tokenize_pairs(inp, tar):
    inp = tokenizer.tokenize(inp)
    inp = add_start_end(inp)
    inp = inp.to_tensor()
    inp = tf.cast(inp, tf.int64)

    tar = tf.strings.to_number(tar, tf.int64)
    return inp, tar

def make_batches(dataset):
    return (dataset
    .cache()
    .shuffle(150000)
    .batch(64)
    .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE))

train_batches = make_batches(train_dataset)
test_batches = make_batches(test_dataset)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads, d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, attention_weight = self.mha(x, x, x, mask, return_attention_scores=True)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attention_weight

class EncoderModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, pe_max, rate=0.1):
        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(pe_max, self.d_model)
        self.dropout = tf.keras.layers.Dropout(rate)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.final_layer = tf.keras.layers.Dense(1000)

    def call(self, x, training):
        mask = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

        seq_len = tf.shape(x)[1]

        attention_weights = list()

        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, attention_weight = self.enc_layers[i](x, training, mask)

            attention_weights.append(attention_weight)

        x = tf.reduce_max(x, axis=1)
        
        out = self.final_layer(x)  # (batch_size, 1000)

        return out, attention_weights

num_layers = 2
d_model = 256
dff = 512
num_heads = 4
dropout_rate = 0.1

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=10000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=1))
    return tf.reduce_mean(tf.cast(accuracies, dtype=tf.float32))

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

encoderModel = EncoderModel(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    vocab_size=tokenizer.vocab_size(),
    pe_max=1000,
    rate=dropout_rate)

ckpt = tf.train.Checkpoint(encoderModel=encoderModel, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, f'{path}/checkpoints/train', max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print('Latest checkpoint restored!!')

# ckpt.restore(ckpt_manager.checkpoints[3]).expect_partial()
# print('Latest checkpoint restored!!')

EPOCHS = 25

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature) # to limit retracing when Tensors have dynamic shapes
def train_step(inp, tar):
    with tf.GradientTape() as tape:
        predictions, _ = encoderModel(inp, training=True)
        loss = loss_function(tar, predictions)

    gradients = tape.gradient(loss, encoderModel.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoderModel.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar, predictions))

# tf.config.run_functions_eagerly(True) # DEBUG

# for epoch in range(EPOCHS):
#     start = time.time()

#     train_loss.reset_states()
#     train_accuracy.reset_states()

#     for (batch, (inp, tar)) in enumerate(train_batches):
#         train_step(inp, tar)

#         if batch % 50 == 0:
#             print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

#     if (epoch + 1) % 5 == 0:
#         ckpt_save_path = ckpt_manager.save()
#         print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

#     print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

#     print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

# acc = list()
# for (batch, (inp, tar)) in enumerate(test_batches):
#     predictions, _ = encoderModel(inp, training=False)

#     accuracy = accuracy_function(tar, predictions).numpy()
#     acc.append(accuracy)

#     print(batch, accuracy)

# print(np.mean(acc))

"""
   2/4/256/512/MA  :   2/4/256/512/MH
0: 0.91371256(20)  :   0.9134326(15)    :   0.91328865(10)
1: 0.9144204(20)   :   0.9138445(15)    :   0.9121489(10)
2: 0.9146684(20)   :   0.9146324(15)    :   0.9135126(10)
3: 0.9149603(20)   :   0.9154642(15)    :   0.9147044(10)
   0.914440         
"""

# 0.9161568(20) :

class Classifier(tf.Module):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, sentence):
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        tokens = add_start_end(self.tokenizer.tokenize(sentence)).to_tensor()

        predictions, attention_weights = self.model(tokens, training=False)

        codes = tf.argmax(predictions, axis=-1)

        return codes, tokens, attention_weights

classifier = Classifier(tokenizer, encoderModel)

def print_translation(sentence, tokens, real, pred):
    tokens = [t.decode('utf-8') + f'({i})' for i, t in enumerate(tokenizer.id_to_string(tokens)[0].numpy())]

    print(f'{"Input":15s}: {sentence}')
    print(f'{"Tokens":15s}: {tokens}')
    print(f'{"Real":15s}: {real}')
    print(f'{"Pred":15s}: {pred[0]}')

def plot_attention_weights(attention_heads):
    fig = plt.figure(figsize=(16, 8))

    a = len(attention_heads)
    b = attention_heads[0].shape[1]

    for i, heads in enumerate(attention_heads):
        for h, head in enumerate(heads[0]):
            ax = fig.add_subplot(a, b, (i*b)+h+1)

            plt.pcolormesh(head, cmap='RdBu')

            ax.set_xlabel(f'L{i} Head {h+1}')

    plt.tight_layout()
    plt.show()

sentence = '치킨전문점에서 고객의주문에의해 치킨판매'
code = -1

codes, tokens, attention_weights = classifier(tf.constant(sentence))
print_translation(sentence, tokens, code, codes)
# plot_attention_weights(attention_weights)

charMap = np.full(100, '')

charMap[:5] = 'A'       # 농업, 임업 및 어업(01~03)
charMap[5:10] = 'B'     # 광업(05~08)
charMap[10:35] = 'C'    # 제조업(10~34)
charMap[35:36] = 'D'    # 전기, 가스, 증기 및 공기 조절 공급업(35)
charMap[36:41] = 'E'    # 수도, 하수 및 폐기물 처리, 원료 재생업(36~39)
charMap[41:45] = 'F'    # 건설업(41~42)
charMap[45:49] = 'G'    # 도매 및 소매업(45~47)
charMap[49:55] = 'H'    # 운수 및 창고업(49~52)
charMap[55:58] = 'I'    # 숙박 및 음식점업(55~56)
charMap[58:64] = 'J'    # 정보통신업(58~63)
charMap[64:68] = 'K'    # 금융 및 보험업(64~66)
charMap[68:70] = 'L'    # 부동산업(68)
charMap[70:74] = 'M'    # 전문, 과학 및 기술 서비스업(70~73)
charMap[74:84] = 'N'    # 사업시설 관리, 사업 지원 및 임대 서비스업(74~76)
charMap[84:85] = 'O'    # 공공 행정, 국방 및 사회보장 행정(84)
charMap[85:86] = 'P'    # 교육 서비스업(85)
charMap[86:90] = 'Q'    # 보건업 및 사회복지 서비스업(86~87)
charMap[90:94] = 'R'    # 예술, 스포츠 및 여가관련 서비스업(90~91)
charMap[94:97] = 'S'    # 협회 및 단체, 수리 및 기타 개인 서비스업(94~96)
charMap[97:99] = 'T'    # 가구 내 고용활동 및 달리 분류되지 않은 자가 소비 생산활동(97~98)
charMap[99:] = 'U'      # 국제 및 외국기관(99)

def make_result():
    df = pd.read_csv(f'{path}/data/result.csv', encoding='cp949')

    sentenceList = list()
    
    for i in df.iloc[:, 4:].to_numpy():
        sentence = list()
        for d in i:
            if type(d) == str:
                sentence.append(d)
        
        sentenceList.append(' '.join(sentence))

    for i in range(len(sentenceList) // 200):
        ids, _, _ = classifier(tf.constant(sentenceList[i*200:(i+1)*200]))

        for j, id in enumerate(ids.numpy()):
            n = id // 10

            df.iat[i*200+j, 1] = charMap[n]
            df.iat[i*200+j, 2] = int(n)
            df.iat[i*200+j, 3] = int(id)

    df.to_csv('result.csv', index=False, encoding='cp949')

# make_result()

def comp_result(fileA, fileB):
    dfA = pd.read_csv(f'{path}/{fileA}.csv', encoding='cp949')
    dfB = pd.read_csv(f'{path}/{fileB}.csv', encoding='cp949')

    dA = dfA.iloc[:, 1:4].to_numpy() 
    dB = dfB.iloc[:, 1:4].to_numpy()
    
    num = 0 

    for i in range(len(dB)):
        cA, cA1, cA2 = dA[i, 0], dA[i, 1], dA[i, 2]
        cB, cB1, cB2 = dB[i, 0], dB[i, 1], dB[i, 2]

        if cA != cB or cA1 != cB1 or cA2 != cB2:
            print(i, cA, cA1, cA2, ':', cB, cB1, cB2, ':', dfA.iloc[i, 4:].to_numpy())
            num += 1

    print(len(dA), num)

comp_result('result_M', 'result_F')

def merge_result(fileList, weight):
    dfs = list()

    for file in fileList:
        dfs.append(pd.read_csv(f'{path}/{file}.csv', encoding='cp949'))

    datas = list()

    for df in dfs:
        datas.append(df.iloc[:, 1:4].to_numpy())
    
    scores = [0 for _ in range(len(fileList))]
    
    num, num_m, num_cm = 0, 0, 0

    df_ref = dfs[0]

    for i in range(len(dfs[0])):
        cB, cB1, cB2 = datas[0][i, 0], datas[0][i, 1], datas[0][i, 2]

        for j in range(len(datas)): # 불일치 확인
            if datas[j][i, 0] != cB or datas[j][i, 1] != cB1 or datas[j][i, 2] != cB2:
                cCDict = dict()
                n1CDict = dict()
                n2CDict = dict()

                reC = cB
                ren1 = int(cB1)
                ren2 = int(cB2)

                for m in range(len(datas)):
                    if datas[m][i, 0] in cCDict:
                        cCDict[datas[m][i, 0]] += 1
                    else:
                        cCDict[datas[m][i, 0]] = 1

                for key, value in cCDict.items():
                    if value > cCDict[reC]:
                        reC = key

                for m in range(len(datas)):
                    if datas[m][i, 0] == reC:
                        if int(datas[m][i, 1]) in n1CDict:
                            n1CDict[int(datas[m][i, 1])] += 1
                        else:
                            n1CDict[int(datas[m][i, 1])] = 1
                
                if not ren1 in n1CDict:
                    ren1 = list(n1CDict.keys())[0]

                for key, value in n1CDict.items():
                    if value > n1CDict[ren1]:
                        ren1 = int(key)

                for m in range(len(datas)): 
                    if int(datas[m][i, 1]) == ren1:   
                        if int(datas[m][i, 2]) in n2CDict:
                            n2CDict[int(datas[m][i, 2])] += 1
                        else:
                            n2CDict[int(datas[m][i, 2])] = 1
                
                if not ren2 in n2CDict:
                    ren2 = list(n2CDict.keys())[0]

                for key, value in n2CDict.items():
                    if value > n2CDict[ren2] + weight:
                        ren2 = int(key)            
                
                if reC != cB or ren1 != cB1 or ren2 != cB2:
                    # print(i, [(datas[g][i, 0], datas[g][i, 1], datas[g][i, 2]) for g in range(len(datas))], '||', reC, ren1, ren2, '||', dfs[0].iloc[i, 4:].to_numpy())
                    
                    if reC != cB:
                        print(i, [(datas[g][i, 0], datas[g][i, 1], datas[g][i, 2]) for g in range(len(datas))], '||', reC, ren1, ren2, '||', dfs[0].iloc[i, 4:].to_numpy())
                        num_cm += 1
                    else:
                        # print(i, [(datas[g][i, 0], datas[g][i, 1], datas[g][i, 2]) for g in range(len(datas))], '||', reC, ren1, ren2, '||', dfs[0].iloc[i, 4:].to_numpy())
                        pass
                    
                    num_m += 1

                for s in range(len(datas)):
                    if datas[s][i, 2] == ren2:
                        scores[s] += 1

                if reC != charMap[ren1] or ren1 != ren2 // 10:
                    raise Exception('ERROR!')
                
                df_ref.iat[i, 1] = reC
                df_ref.iat[i, 2] = ren1
                df_ref.iat[i, 3] = ren2
                
                num += 1
                break
    
    df_ref.to_csv('result_M.csv', index=False, encoding='cp949')

    print(len(dfs[0]), num, num_m, num_cm)
    print(scores)

# merge_result(['result_F', 'result_A', 'result_B', 'result_C', 'result_D'], 0)
