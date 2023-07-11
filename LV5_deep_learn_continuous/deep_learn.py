from tensorflow import keras
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display


red = pd.read_csv('red-wine.csv')

dtrain = red.sample(frac=0.7, random_state=0)
dvalid = red.drop(dtrain.index)

max_train = dtrain.max(axis=0)
min_train = dtrain.min(axis=0)
# Min-max scaling
dtrain = (dtrain - min_train)/(max_train - min_train)
dvalid = (dvalid - min_train)/(max_train - min_train)

xtrain = dtrain.drop('quality', axis=1)
xvalid = dvalid.drop('quality', axis=1)
ytrain = dtrain['quality']
yvalid = dvalid['quality']

# 提早停止
# "If there hasn't been at least an improvement of 0.001 in the validation loss
# over the previous 20 epochs, then stop the training and keep the best model you found."
early = callbacks.EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True,
)


model = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=[11]),
    layers.Dropout(0.4),    # 隨機隱藏部分神經元
    layers.BatchNormalization(),   # Normalization

    layers.Dense(512, activation='relu'),
    layers.Dropout(0.4),
    layers.BatchNormalization(),

    layers.Dense(512, activation='relu'),
    layers.Dropout(0.4),
    layers.BatchNormalization(),

    layers.Dense(1),
])

model.compile(
    optimizer='adam',
    loss='mae'
)

history = model.fit(
    xtrain, ytrain,
    validation_data=(xvalid, yvalid),
    batch_size=256,
    epochs=500,
    callbacks=[early],
    verbose=0  # turn off
)

hisd = pd.DataFrame(history.history)
print(hisd['val_loss'].min())
hisd.loc[:, ['loss', 'val_loss']].plot()
plt.show()
