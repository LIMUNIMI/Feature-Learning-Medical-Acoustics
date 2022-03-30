import leaf_audio
import tensorflow as tf
from leaf_audio import frontend
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Any
import functools
import pandas as pd
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

##### GPU #####
device_name = tf.test.gpu_device_name()
if not device_name:
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64
EPOCHS = 300
LEARN_RATE = 1e-5



def createDS():
    path = '../PHYSIONET_HeartBeat_SEGMENTED/'
    frames = 4000
    
    df = pd.read_csv(path+'Reference_total.csv')
    label = df['class_label']
    file_path = df['name']
    
    data = []
    for x in range(len(file_path)):
        filename = path+file_path[x]
        raw_audio = tf.io.read_file(filename)
        tf_audio, _ = tf.audio.decode_wav(raw_audio, desired_samples=frames)
        audio_sample = tf.squeeze(tf_audio, axis=-1)
        audio_sample = audio_sample[tf.newaxis, :]
        data.append(audio_sample)

    
    print("Dimensione ds--> ", len(data))
    
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10


    train_data, test_data, train_label, test_label = train_test_split(data,label, test_size=1 - train_ratio, random_state=101, shuffle=True)
    

    val_data, test_data, val_label, test_label = train_test_split(test_data, test_label, test_size=test_ratio/(test_ratio + validation_ratio), random_state=101, shuffle=True) 

    
    
    train_data = tf.convert_to_tensor(train_data)
    val_data = tf.convert_to_tensor(val_data)
    test_data = tf.convert_to_tensor(test_data)

    train_data = tf.transpose(train_data, perm=[0,2,1])
    val_data = tf.transpose(val_data, perm=[0,2,1])
    test_data = tf.transpose(test_data, perm=[0,2,1])
    
    
    print("Dimensione train_ds  ---> ", train_data.shape) 
    print("Dimensione valitation_ds  ---> ", val_data.shape)
    print("Dimensione test_ds  ---> ", test_data.shape)

    
    train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    val_ds = tf.data.Dataset.from_tensor_slices((val_data, val_label))
    test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_label))
    

    print("DATASET DONE")
    return train_ds, val_ds, test_ds






class MIOClassifier(tf.keras.Model):

  def __init__(self,
               num_outputs: int):

    super().__init__()
    self._frontend = frontend.MelFilterbanks(n_filters=128,
                                  window_len=30.,
				  window_stride=10.,
                                  sample_rate=2000,
                                  min_freq=25.,
                                  max_freq=1000.)

    self._resize = tf.keras.layers.Resizing(224, 224)
    self._effNet = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=(224,224,1), drop_connect_rate=0.5, classes=2)
        
    self._pool = tf.keras.layers.Flatten()
    self._head = tf.keras.layers.Dense(num_outputs, activation=tf.keras.activations.softmax) 

  def call(self, inputs: tf.Tensor):
        output = tf.squeeze(inputs)

        output = self._frontend(output) #(batch_size, time_frames, freq_bins)

        output = tf.transpose(output, perm=[1,2,0]) 
        output = self._resize(output) 
        output = tf.transpose(output, perm=[2,0,1]) 
        output = tf.expand_dims(output, -1) 

        output = tf.keras.applications.efficientnet.preprocess_input(output)
        output = self._effNet(output)

        output = self._pool(output)
        return self._head(output)


def build_model():
    mioModel = MIOClassifier(num_outputs=2)

    return mioModel





def train(num_epochs: int = EPOCHS,
          learning_rate: float = LEARN_RATE,
          batch_size: int = BATCH_SIZE,
          train_ds = None,
          val_ds = None,
          test_ds = None,
          model = None,
          **kwargs):

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy() 
    metric = 'sparse_categorical_accuracy'  
    model.compile(loss=loss_fn,
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                metrics=[metric])

    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./log/Leaf', histogram_freq=1)

    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    
    storia = model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, shuffle=True, callbacks=[CustomCallback(model, test_ds)])
    
    model.summary()

    return storia






def train_model(model):
    train_ds, val_ds, test_ds = createDS()

    f = open("cuore_eff_melspec.txt","w+")
    f.write("Cuore Efficientnet MelSpec \n")
    f.close()

    storia = train(train_ds=train_ds, val_ds=val_ds, model=model, test_ds=test_ds)
  

    # evaluate the model
    eval_test_ds = test_ds.batch(BATCH_SIZE)
    test_acc = model.evaluate(eval_test_ds, verbose=0)
    print("Evaluate test loss, test acc:", test_acc)
    print("---------------------")


    metrics = storia.history
    plt.plot(storia.epoch, metrics['sparse_categorical_accuracy'], metrics['val_sparse_categorical_accuracy'])
    plt.legend(['accuracy', 'val_accuracy'])
    plt.title("Accuracy", size=16)
    plt.savefig('accmelSpec.png')

    plt.plot(storia.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.title("Loss", size=16)
    plt.savefig('lossmelSpec.png')


    return test_ds





def test_model(test_ds, model):

    test_audio = []
    lables = []

    for test_data, test_label in test_ds.take(-1): 
        test_audio.append(test_data)
        lables.append(test_label.numpy())
        
    test_audio = tf.convert_to_tensor(test_audio)


    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = lables



    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Final test set accuracy: {test_acc:.2%}')



    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_true, y_pred)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_true, y_pred)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_true, y_pred)
    print('Recall: %f' % recall)
    print("Balanced Accuracy", balanced_accuracy_score(y_true, y_pred))

    print(confusion_matrix(y_true, y_pred))

    ###### CONFUSION MATRIX #########
    '''class_names = ['Normal', 'Abnormal']
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
                xticklabels=class_names,
                yticklabels=class_names,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')'''
    #plt.savefig("melSpecCM.png")



class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, test_ds):
        self.model = model
        self.test_ds = test_ds
        

    def on_epoch_end(self, epoch, logs={}):
        test_audio = []
        lables = []

        for test_data, test_label in self.test_ds.take(-1): 
            test_audio.append(test_data)
            lables.append(test_label.numpy())
        
        test_audio = tf.convert_to_tensor(test_audio)
        y_pred = np.argmax(self.model.predict(test_audio), axis=1)
        y_true = lables

        test_acc = sum(y_pred == y_true) / len(y_true)
        print("\n")
        print(f'Test set accuracy for epoch {epoch}: {test_acc:.2%}')
        print("BALANCED accuracy: ", balanced_accuracy_score(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        

        f = open("cuore_eff_melspec.txt","a+")
        f.write("*********** Predizione **************** Epoca: %d \n" % epoch)
        for num in y_pred:
            f.write(str(num))
            f.write(", ") 
        f.write("\n")
        f.close()
