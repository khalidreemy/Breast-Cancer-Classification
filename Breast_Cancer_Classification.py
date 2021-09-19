import numpy as np
import keras
from keras import backend


from keras.metrics import categorical_crossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import plot_confusion_matrix as pcm
import itertools
import matplotlib.pyplot as plt

from keras.applications.vgg19 import VGG19

#This function is used to plot images
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

        
from google.colab import drive
import os
drive.mount('/content/drive')


#importing images and performing augmentation using keras ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
#train_path = '/content/drive/My Drive/DataSet/MC/Train'
#valid_path = '/content/drive/My Drive/DataSet/MC/Valid'
train_path = '/content/drive/My Drive/DataSet/CM pics/Train'
valid_path = '/content/drive/My Drive/DataSet/CM pics/Valid'

train_batches = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   rotation_range = 45
                                   ).flow_from_directory(train_path,target_size=(224,224), classes=['Calcification','Mass'], batch_size=32)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path,target_size=(224,224), classes=['Calcification','Mass'], batch_size=32)
# Input example
import matplotlib.pyplot as plt
from keras.preprocessing import image

img = train_batches[750]
plt.figure()
plt.imshow(img)
#Plot image is saved in /images/sample.png

#Building the model using VGG16
from keras.applications import VGG16
from keras import layers
from keras import models
from keras import optimizers


vgg_model = VGG16(weights='imagenet', include_top = False, input_shape =(224,224, 3))

for layer in vgg_model.layers[:0]:
    layer.trainable = True
for layer in vgg_model.layers[0:3]:
    layer.trainable = False
for layer in vgg_model.layers[3:]:
    layer.trainable = True

model = models.Sequential()
model.add(vgg_model)
model.add(layers.Flatten())
model.add(layers.Dropout(0.4))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(2, activation='sigmoid', name='softmax'))

#model.summary()
model.compile( optimizer=optimizers.Adam(lr=4e-5),
#     optimizer=optimizers.RMSprop(lr=7e-5),
#    loss='sparse_categorical_crossentropy', 
    loss='categorical_crossentropy', 
    metrics=['accuracy'])

#fitting the model
model.fit_generator(train_batches, epochs=30, verbose=1, validation_data=valid_batches, callbacks=[ReduceLROnPlateau(patience=5)])


history = net_final.fit_generator(train_iterator, 
                              steps_per_epoch=train_samples // BATCH_SIZE, 
                              epochs=NUM_EPOCHS,
                              validation_data=validation_iterator,
                              validation_steps=valid_samples // BATCH_SIZE)

## Epoch 1/30 64/64 [==============================] - 22s 340ms/step - loss: 0.6590 - acc: 0.6344 - val_loss: 0.4706 - val_acc: 0.8125
## Epoch 2/30 64/64 [==============================] - 22s 343ms/step - loss: 0.4735 - acc: 0.7624 - val_loss: 0.3397 - val_acc: 0.8698
## Epoch 3/30 64/64 [==============================] - 22s 339ms/step - loss: 0.3496 - acc: 0.8627 - val_loss: 0.3121 - val_acc: 0.8574
## Epoch 4/30 64/64 [==============================] - 21s 335ms/step - loss: 0.3102 - acc: 0.8729 - val_loss: 0.3798 - val_acc: 0.8285
## Epoch 5/30 64/64 [==============================] - 21s 330ms/step - loss: 0.2641 - acc: 0.8944 - val_loss: 0.3571 - val_acc: 0.8554
## Epoch 6/30 64/64 [==============================] - 21s 327ms/step - loss: 0.2530 - acc: 0.8945 - val_loss: 0.2926 - val_acc: 0.8926
## Epoch 7/30 64/64 [==============================] - 21s 324ms/step - loss: 0.2681 - acc: 0.8963 - val_loss: 0.2954 - val_acc: 0.9050
## Epoch 8/30 64/64 [==============================] - 21s 324ms/step - loss: 0.2254 - acc: 0.9159 - val_loss: 0.2756 - val_acc: 0.8802
## Epoch 9/30 64/64 [==============================] - 21s 325ms/step - loss: 0.2171 - acc: 0.9134 - val_loss: 0.2779 - val_acc: 0.9070
## Epoch 10/30 64/64 [==============================] - 21s 329ms/step - loss: 0.2218 - acc: 0.9081 - val_loss: 0.2987 - val_acc: 0.9008
## Epoch 11/30 64/64 [==============================] - 21s 322ms/step - loss: 0.1912 - acc: 0.9335 - val_loss: 0.2151 - val_acc: 0.9339
## Epoch 12/30 64/64 [==============================] - 21s 321ms/step - loss: 0.1937 - acc: 0.9258 - val_loss: 0.1971 - val_acc: 0.9277
## Epoch 13/30 64/64 [==============================] - 21s 324ms/step - loss: 0.1890 - acc: 0.9330 - val_loss: 0.3340 - val_acc: 0.8967
## Epoch 14/30 64/64 [==============================] - 21s 321ms/step - loss: 0.1597 - acc: 0.9380 - val_loss: 0.2262 - val_acc: 0.9153
## Epoch 15/30 64/64 [==============================] - 21s 323ms/step - loss: 0.1621 - acc: 0.9433 - val_loss: 0.3053 - val_acc: 0.9008
## Epoch 16/30 64/64 [==============================] - 21s 327ms/step - loss: 0.1531 - acc: 0.9424 - val_loss: 0.2503 - val_acc: 0.9339
## Epoch 17/30 64/64 [==============================] - 21s 324ms/step - loss: 0.1654 - acc: 0.9414 - val_loss: 0.3026 - val_acc: 0.9153
## Epoch 18/30 64/64 [==============================] - 21s 324ms/step - loss: 0.1873 - acc: 0.9170 - val_loss: 0.3271 - val_acc: 0.9180
## Epoch 19/30 64/64 [==============================] - 21s 321ms/step - loss: 0.1583 - acc: 0.9413 - val_loss: 0.2146 - val_acc: 0.9339
## Epoch 20/30 64/64 [==============================] - 20s 316ms/step - loss: 0.1242 - acc: 0.9526 - val_loss: 0.3942 - val_acc: 0.8926
## Epoch 21/30 64/64 [==============================] - 21s 321ms/step - loss: 0.1309 - acc: 0.9512 - val_loss: 0.2774 - val_acc: 0.9236
## Epoch 22/30 64/64 [==============================] - 20s 318ms/step - loss: 0.1357 - acc: 0.9482 - val_loss: 0.2399 - val_acc: 0.9256
## Epoch 23/30 64/64 [==============================] - 20s 319ms/step - loss: 0.1111 - acc: 0.9570 - val_loss: 0.2504 - val_acc: 0.9380
## Epoch 24/30 64/64 [==============================] - 21s 321ms/step - loss: 0.1266 - acc: 0.9561 - val_loss: 0.2548 - val_acc: 0.9277
## Epoch 25/30 64/64 [==============================] - 20s 316ms/step - loss: 0.1285 - acc: 0.9517 - val_loss: 0.3120 - val_acc: 0.8884
## Epoch 26/30 64/64 [==============================] - 20s 313ms/step - loss: 0.1258 - acc: 0.9504 - val_loss: 0.2652 - val_acc: 0.9298
## Epoch 27/30 64/64 [==============================] - 21s 324ms/step - loss: 0.1042 - acc: 0.9614 - val_loss: 0.3332 - val_acc: 0.9070
## Epoch 28/30 64/64 [==============================] - 20s 316ms/step - loss: 0.1142 - acc: 0.9559 - val_loss: 0.3010 - val_acc: 0.9008
## Epoch 29/30 64/64 [==============================] - 21s 324ms/step - loss: 0.1216 - acc: 0.9560 - val_loss: 0.2630 - val_acc: 0.9360
## Epoch 30/30 64/64 [==============================] - 21s 321ms/step - loss: 0.0912 - acc: 0.9648 - val_loss: 0.2457 - val_acc: 0.9360

# plot the history
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.ylim((0,1))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.ylim((0,1))

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#Plot image is saved in /images/plot1.png


# plot the history
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.ylim((0,1))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.ylim((0,1))

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#Plot image is saved in /images/plot2.png


results = net_final.evaluate_generator(test_iterator, steps=22, verbose=1)

# #doesn't work
probabilities = net_final.predict(test_rgb_imgs)
print("test_loss: ", results[0], " test_acc: ", results[1])

results = net_final.evaluate_generator(test_iterator, steps=22, verbose=1)

## 22/22 [==============================] - 2s 78ms/step
## test_loss:  0.19931246756567997  test_acc:  0.9498567336951422
