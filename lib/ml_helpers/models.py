import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lib.ml_helpers import custom_loss, moa
from lib.utils import array_trimmer


class TestingGenerator(tf.keras.utils.Sequence):
    """
    A generator to process testing data iteratively.
    """

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.s0, self.s1 = self.x.shape[1], self.x.shape[2]
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


class TrainingGenerator(TestingGenerator):
    """
    A generator to feed training data iteratively, with augmentation.
    """

    def __init__(self, x_set, y_set, batch_size):
        super().__init__(x_set, y_set, batch_size)

    @tf.function
    def data_augmentation(self, batch_x, batch_y):
        """
        Augments input data by rolling an input pattern randomly and changing
        its relative intensity.
        """

        def roll_drp(inputs):
            # (i) Roll reflectance pattern
            reflectance_patterns, eulers = inputs
            reflectance_patterns = tf.expand_dims(reflectance_patterns, axis=0)
            idx_shift = tf.random.uniform(shape=(1,),
                                          maxval=self.s1,
                                          dtype=tf.dtypes.int32)
            angular_shift = idx_shift / self.s1 * np.pi * 2
            reflectance_patterns = tf.roll(reflectance_patterns, shift=idx_shift[0], axis=2)
            eulers = tf.stack((
                tf.add(eulers[0], angular_shift[0]),
                eulers[1],
                eulers[2],
            ), axis=0)

            # (ii) Modify relative intensity based on normal distribution
            reflectance_patterns = tf.multiply(reflectance_patterns, tf.random.normal(
                shape=(1,),
                mean=1.0,
                stddev=0.2,
                dtype=tf.dtypes.float64))
            reflectance_patterns = tf.clip_by_value(reflectance_patterns, 0, 255)
            reflectance_patterns = tf.squeeze(reflectance_patterns)

            return reflectance_patterns, eulers

        # Apply augmentation function and yield augmented batch
        batch_x, batch_y = tf.map_fn(
            fn=roll_drp,
            elems=(batch_x, batch_y),
            parallel_iterations=len(batch_x),
        )

        return batch_x, batch_y

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x, batch_y = self.data_augmentation(batch_x, batch_y)

        return batch_x, batch_y


class CustomModel:
    def __init__(self, root, bs, test_size, s0, s1, checkpoint, train_samples, test_samples):

        self.bs = bs
        self.root = root
        self.s0, self.s1 = s0, s1
        self.checkpoint = checkpoint
        self.total_epochs = 0
        self.lr = 0

        # Get training dataset
        self.training_dico, self.xtr, self.ytr = self.get_dataset(
            train_samples, subset='training_sets')
        
        # Keep a fraction of the data as validation set
        self.xtr, self.xval, self.ytr, self.yval = train_test_split(
            self.xtr, self.ytr, test_size=test_size)

        # Trim away last batch
        self.xtr, self.xval, self.ytr, self.yval = array_trimmer(
            self.bs, self.xtr, self.xval, self.ytr, self.yval)

        # Get testing dataset
        self.testing_dico, self.xte, self.yte = self.get_dataset(
            test_samples, subset='evaluation_sets')

        self.xte, self.yte = array_trimmer(self.bs, self.xte, self.yte)

        print()
        print('Training data: ', self.xtr.shape, self.ytr.shape)
        print('Test data: ', self.xte.shape, self.yte.shape)

        self.validation_dataset = TestingGenerator(self.xval, self.yval, self.bs)
        self.testing_dataset = TestingGenerator(self.xte, self.yte, self.bs)

        self.kernel_neurons_1 = None
        self.kernel_neurons_2 = None
        self.dense_neurons_1 = None
        self.dense_neurons_2 = None
        self.model = None

    def get_dataset(self, sample_names, subset):
        """
        Loads data from selected sample names.
        """
        xtr = np.empty((0, self.s0, self.s1))
        ytr = np.empty((0, 3))
        big_d = {}
        for name in sample_names:
            path = os.path.join(self.root, f'{subset}/{name}.npy')
            dataset = np.load(path, allow_pickle=True).item()
            xtr_sample = dataset.get('xtr')
            ytr_sample = dataset.get('ytr')
            big_d[name] = (xtr_sample, ytr_sample)
            xtr = np.vstack((xtr, xtr_sample))
            ytr = np.vstack((ytr, ytr_sample))
        idx = np.arange(len(xtr))
        np.random.shuffle(idx)
        xtr = xtr[idx]
        ytr = ytr[idx]

        return big_d, xtr, ytr

    def create_model(self, kernel_neurons_1=16, kenerl_neurons_2=16, dense_neurons_1=64, dense_neurons_2=32):

        self.kernel_neurons_1 = kernel_neurons_1
        self.kernel_neurons_2 = kenerl_neurons_2
        self.dense_neurons_1 = dense_neurons_1
        self.dense_neurons_2 = dense_neurons_2

        @tf.function
        def preprocess(x):
            """
            Rescales input to the range 0-1.
            """
            x = tf.multiply(x, 1 / 255.0)
            x = tf.expand_dims(x, axis=-1)
            return x

        inputs = tf.keras.Input(shape=(self.s0, self.s1))

        # Preprocessing layer (rescales input to 0-1)
        lambds = tf.keras.layers.Lambda(
            lambda x: preprocess(x), output_shape=(self.s0, self.s1, 2))(inputs)

        # Define the output
        output = self.euler_net(lambds)  # EulerNet
        # output = self.reference_model(lambds)  # Jha et al.

        # Build the model
        self.model = tf.keras.Model(inputs=inputs, outputs=output)

        # Compile the model
        self._compile()
        print(self.model.summary())

    def _compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=custom_loss,
            metrics=[moa],
        )
        
    def euler_net(self, lambds):
        """
        Core EulerNet architecture.
        """
        init = tf.keras.initializers.VarianceScaling(0.5)

        conv2d0 = tf.keras.layers.Conv2D(self.kernel_neurons_1, (3, 3), strides=(1, 1), padding="same",
                                         activation='relu')(lambds)
        mpool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv2d0)
        conv2d1 = tf.keras.layers.Conv2D(self.kernel_neurons_2, (3, 3), strides=(1, 1), padding="same",
                                         activation='relu')(mpool1)
        mpool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2d1)

        globav = tf.keras.layers.Flatten()(mpool2)

        dense0 = tf.keras.layers.Dense(self.dense_neurons_1 * 3, activation='relu', kernel_initializer=init)(globav)
        dense01 = tf.keras.layers.Dense(self.dense_neurons_2 * 3, activation='relu', kernel_initializer=init)(dense0)
        output = tf.keras.layers.Dense(3, activation=None)(dense01)

        return output

    # def reference_model(self, lambds):
    #     """
    #     Deeper model architecture - Does not lead to any improvement.
    #     """
    #     init = tf.keras.initializers.VarianceScaling(0.5)

    #     conv2d00 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same",
    #                                      activation='relu')(lambds)
    #     conv2d01 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same",
    #                                      activation='relu')(conv2d00)
    #     conv2d02 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding="same",
    #                                      activation='relu')(conv2d01)
    #     conv2d03 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding="same",
    #                                      activation='relu')(conv2d02)
        
    #     mpool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv2d03)
        
    #     conv2d10 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same",
    #                                      activation='relu')(mpool1)
    #     conv2d11 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same",
    #                                      activation='relu')(conv2d10)
    #     conv2d12 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding="same",
    #                                      activation='relu')(conv2d11)
    #     conv2d13 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding="same",
    #                                      activation='relu')(conv2d12)
    #     mpool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2d13)

    #     globav = tf.keras.layers.Flatten()(mpool2)

    #     dense0 = tf.keras.layers.Dense(512, activation='relu', kernel_initializer=init)(globav)
    #     dense1 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=init)(dense0)
        
    #     output = tf.keras.layers.Dense(3, activation=None)(dense1)

    #     return output

    def run_lr_scheduler(self):
        """
        Learning rate optimization between lr=1e-6 and 1e-2 (50 epochs)
        """
        model = self.model

        training_dataset = TrainingGenerator(
            self.xtr, self.ytr, self.bs)

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 1e-5 * 10 ** (epoch / 10))

        history = model.fit(
            training_dataset,
            epochs=40,
            verbose=1,
            validation_data=self.validation_dataset,
            callbacks=[lr_scheduler],
        )

        lrs = history.history["lr"]
        losses = history.history["loss"]
        optimal_lr = lrs[np.argmin(losses)]

        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
        ax.semilogx(lrs, losses)
        ax.set_xlabel('learning rate')
        ax.set_ylabel('loss')
        ax.set_title('Optimal lr: {:.2e}'.format(optimal_lr))
        plt.show()

        self.create_model(
            self.kernel_neurons_1,
            self.kernel_neurons_2,
            self.dense_neurons_1,
            self.dense_neurons_2,
        )

        return optimal_lr  # Optimal learning rate!

    def set_learning_rate(self, lr):
        K.set_value(self.model.optimizer.learning_rate, lr)
        print('\n> Set learning rate to: {:.2e}'.format(lr))
        self.lr = lr

    def fit(self, patience=None, epochs=1, save_it=False):
        """
        Fits the model, until validation stabilizes for [patience] epochs
        (if specified) or after a pre-determined number of epochs otherwise.
        """
        callbacks = []

        if patience is not None:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=patience)
            callbacks.append(early_stopping)

        if save_it:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint, save_weights_only=True, verbose=1)
            callbacks.append(cp_callback)

        training_dataset = TrainingGenerator(self.xtr, self.ytr, self.bs)

        history = self.model.fit(
            training_dataset,
            epochs=epochs,
            verbose=1,
            validation_data=self.validation_dataset,
            callbacks=callbacks,
        )

        fig, ax = plt.subplots(figsize=(16, 6))
        for metric, values in history.history.items():
            ax.plot(values, label=metric)
        ax.set_ylim(bottom=0)
        plt.legend()
        plt.show()

        self.total_epochs = len(history.history['loss'])
        print('FINISHED - TOTAL EPOCHS: ', self.total_epochs)

    def reload(self, checkpoint):
        """Reloads a model from checkpoint."""
        self.model.load_weights(checkpoint).expect_partial()
        self._compile()
        print(f'\n> Reloaded model weights from: {checkpoint}')

    def predict(self, dataset):
        return self.model.predict(dataset)

    # def evaluate(self, on='test'):
    #     """
    #     Computes MOA histograms and averages over training, val or test set.
    #     """
    #     if on == 'training':
    #         dataset = TestingGenerator(self.xtr, self.ytr, self.bs)
    #         preds = self.predict(dataset)
    #         moas, median_moa, ma_x, ma_y, ma_z = evaluate(preds,  self.ytr)
    #     elif on == 'test':
    #         dataset = self.testing_dataset
    #         preds = self.predict(dataset)
    #         moas, median_moa, ma_x, ma_y, ma_z = evaluate(preds, self.yte)
    #     else:
    #         dataset = self.validation_dataset
    #         preds = self.predict(dataset)
    #         moas, median_moa, ma_x, ma_y, ma_z = evaluate(preds, self.yval)
        
    #     return moas, median_moa, ma_x, ma_y, ma_z
