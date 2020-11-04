import wandb
from wandb.keras import WandbCallback
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras import layers, models, optimizers

# Hyperparameters
defaults=dict(
    # Preprocessing

    # Model 
    epochs = 5,
    batch_size = 32,
    conv_layer_1 = 32,
    conv_layer_2 = 64,
    conv_layer_3 = 64,
    fc_layer_1 = 64,
    learning_rate = 1e-3,
    optimizer = 'Adam'
)
wandb.init(config=defaults, resume=True, name='Second Model', project='NN_Project_Test_Runs', notes='more epochs and use Adam opt')
config = wandb.config

# Load dataset as dataframe
train_df = pd.read_csv("txt_files/gender_train.txt", sep=' ', names=['datadir', 'gender'])
test_df = pd.read_csv("txt_files/gender_test.txt", sep=' ', names=['datadir', 'gender'])
train_df['datadir'] = 'data/aligned/' + train_df['datadir'].astype(str)
test_df['datadir'] = 'data/aligned/' + test_df['datadir'].astype(str)

# Load images into keras image generator 
datagen_train = ImageDataGenerator(rescale=1./255)
datagen_valid = ImageDataGenerator(rescale=1./255)
# For train generator
train_generator = datagen_train.flow_from_dataframe(
    dataframe = train_df,
    directory=None,
    x_col="datadir",
    y_col="gender",
    batch_size=config.batch_size,
    seed=42,
    shuffle=True,
    class_mode='raw',
    target_size=(256,256))
# For test generator 
valid_generator = datagen_valid.flow_from_dataframe(
    dataframe = test_df,
    directory=None,
    x_col="datadir",
    y_col="gender",
    batch_size=config.batch_size,
    seed=42,
    shuffle=True,
    class_mode='raw',
    target_size=(256,256))

# Define CNN model 
model = models.Sequential()
model.add(layers.Conv2D(config.conv_layer_1, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(config.conv_layer_2, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(config.conv_layer_3, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(config.fc_layer_1, activation='relu'))
model.add(layers.Dense(1,activation='softmax'))

# Set optimizer for wandb
if(config.optimizer == 'SGD'):
    SGD = optimizers.SGD(learning_rate=config.learning_rate)
    optimizer = SGD
elif(config.optimizer == 'Adam'):
    Adam = optimizers.Adam(learning_rate=config.learning_rate)
    optimizer = Adam
else:
    RMSprop = optimizers.RMSprop(learning_rate=config.learning_rate)
    optimizer = RMSprop

# Compile model 
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

# Fit model and save weights
model.fit(train_generator, epochs=config.epochs, validation_data=valid_generator, callbacks=[WandbCallback()])
model.save_weights('second_model.h5') 