import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras import Model


class MyModel(Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(16,3, data_format ='channels_last', activation='relu') # Nx28x28x16
        self.conv2 = Conv2D(32,3, data_format ='channels_last', activation='relu') # Nx28x28x32
        self.d1 = Dense(10, activation='softmax')

    def call(self, inputs):
        print('initial dim = ', inputs.shape)
        x = self.conv1(inputs)
        print('After conv1 = ', x.shape)
        x = self.conv2(x)
        print('After conv2 = ', x.shape)
        x = self.d1(x)
        print('After d1 = ', x.shape)
        return x


def simple_method_train(x_train, y_train, x_test, y_test):
    #build a model
    model = tf.keras.models.Sequential([
                    Flatten(input_shape=(28, 28)),
                    Dense(128, activation='relu'),
                    Dropout(0.2),
                    Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.summary()
    model.fit(x_train, y_train, epochs=20)
    model.evaluate(x_test, y_test, verbose=2)

if __name__ == "__main__":

    #from tensorflow.python.client import device_lib
    #print(device_lib.list_local_devices())
    print(tf.test.is_gpu_available())
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    #exit()
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)
    
    #normalize data
    x_train = x_train/255.0
    x_test = x_test/255.0
    #a simple method using Dense layers 
    #simple_method_train(x_train, y_train, x_test, y_test)
    
    # a simple method for conv2D layers

    #data_format='channels_last'.
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    print(x_train.shape, x_test.shape)

    model = MyModel()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()


    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=20)
    model.evaluate(x_test, y_test, verbose=2)


