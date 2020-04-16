# Modeling of the U-Net

import tensorflow as tf

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 2

inputs= tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
C1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
C1 = tf.keras.layers.Dropout(0.1)(C1)
C1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(C1)
P1 = tf.keras.layers.MaxPooling2D((2,2))(C1)

C2 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(P1)
C2 = tf.keras.layers.Dropout(0.1)(C2)
C2 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(C2)
P2 = tf.keras.layers.MaxPooling2D((2,2))(C2)

C3 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(P2)
C3 = tf.keras.layers.Dropout(0.1)(C3)
C3 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(C3)
P3 = tf.keras.layers.MaxPooling2D((2,2))(C3)

C4 = tf.keras.layers.Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(P3)
C4 = tf.keras.layers.Dropout(0.1)(C4)
C4 = tf.keras.layers.Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(C4)
P4 = tf.keras.layers.MaxPooling2D((2,2))(C4)

C5 = tf.keras.layers.Conv2D(1024, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(P4)
C5 = tf.keras.layers.Dropout(0.1)(C5)
C5 = tf.keras.layers.Conv2D(1024, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(C5)


# Expansion
U6 = tf.keras.layers.Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(C5)
U6 = tf.keras.layers.concatenate([U6,C4])
C6 = tf.keras.layers.Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(U6)
C6 = tf.keras.layers.Dropout(0.1)(C6)
C6 = tf.keras.layers.Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(C6)

U7 = tf.keras.layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(C6)
U7 = tf.keras.layers.concatenate([U7,C3])
C7 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(U7)
C7 = tf.keras.layers.Dropout(0.1)(C6)
C7 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(C7)

U8 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(C7)
U8 = tf.keras.layers.concatenate([U8,C2])
C8 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(U8)
C8 = tf.keras.layers.Dropout(0.1)(C8)
C8 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(C8)

U9 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(C8)
U9 = tf.keras.layers.concatenate([U9,C1], axis = 2)
C9 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(U9)
C9 = tf.keras.layers.Dropout(0.1)(C9)
C9 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(C9)

outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(C9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#Checkpointer
checkpointer = tf.keras.callbacks.ModelCheckpoint('')

callbacks = 
    [    tf.keras.callbacks.EarlyStopping(patience=2,monitor='val_loss')
         tf.keras.callbacks.TensorBoard(log_dir='logs')]
    
results = model.fit()
