from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from weight_init import leastsq_init, eye_init
import matplotlib.pyplot as plt
from keras.optimizers import Adam, SGD
import time
import numpy as np
import pdb


class Stacked_AEC:
    ''' This is an implementation of layer-wise training for a stacked auto-encoder'''

    def __init__(self, save_dir, mode = 'Identity', save_wts = False):
        self.weights = []
        self.values = []
        self.pred_val = []
        self.tr_error = []
        self.SAVE_WTS = save_wts
        self.mode = mode
        self.dir_name = save_dir
        self._mkdirs()

    def _mkdirs(self):
        try:
            print('Creating dir {}'.format(self.dir_name))
            os.mkdir(self.dir_name)
        except:
            print('{} already exist'.format(self.dir_name))

    def fit(self, train_X, train_y, test_X, test_y, depth, epochs, lr_rate, batch_size=100, opti='SGD', loss='mse'):
        INPUT = train_X.shape[1]
        HID = OUTPUT = train_y.shape[1]
        self.lr_rate = lr_rate
        self.epochs = epochs

        if opti == 'SGD':
            # default == SGD
            opt = SGD(lr=self.lr_rate, decay=0.0, momentum=0.9, nesterov=False)
        elif opti == 'Adam':
            opt = Adam(lr=self.lr_rate)
        else:
            # Custom passed optimizer from outside
            opt = opti

        # RUN TRAINING LOOP OVER THE DEPTH
        weights = []
        start_time = time.time()
        for i in range(depth):
            print('Starting training NN with {} layer...'.format(i+1))
            if i==0:
                model = self.createModel(INPUT,HID, i+1)
                wts = self.init_fun(self.mode, INPUT, HID, train_X, train_y, None)
                weights.append(wts[0])
                weights.append(wts[1])
                model.set_weights(weights) # IDENTIT
                
                model.compile(loss=loss, optimizer=opt) 
                
                hist = model.fit(train_X, train_y, epochs=self.epochs, batch_size=batch_size, validation_data=(test_X, test_y)) 
                # STORE PREDICTED VALUES FROM FIRST NETWORK
                self.values.append(model.predict(train_X, batch_size=batch_size))

            elif i > 0:
                model = self.createModel(INPUT,HID, i+1)
                #SET the other wts
                wts = self.init_fun(self.mode, INPUT, HID, train_X, train_y, self.values[i-1])
                weights.append(wts[0])
                weights.append(wts[1])
                # SET all wts at once
                model.set_weights(weights)

                # opt = SGD(lr=self.lr_rate, decay=0.0, momentum=0.9, nesterov=False) 
                model.compile(loss=loss, optimizer=opt) 
                
                # FIT THE NETWORK WITH THE PREVIOUSLY PREDICTED OUTPUT VALUES
                hist = model.fit(train_X, train_y, epochs=self.epochs, batch_size=batch_size, validation_data=(test_X,test_y))

                self.values.append(model.predict(self.values[i-1], batch_size=batch_size))
            ########################################################
            weights = model.get_weights()
            self.tr_error.append(hist.history)
        # FINE-TUNE THE FINALS LAYERS
        print('{} layers are trained, adding final layer and fine-tuning all the {} layers'.format(depth, depth+1))
        model = self.createModel(INPUT,HID, depth+1)
        wts = self.init_fun(self.mode, INPUT, HID, train_X, train_y, self.values[i-1])
        # wts = self.init_fun(self.mode, INPUT, HID, train_X, train_y, None)

        weights.append(wts[0])
        weights.append(wts[1])
        # SET all wts at once
        model.set_weights(weights)
        model.compile(loss=loss, optimizer=opt) 

        hist = model.fit(train_X, train_y, epochs=self.epochs, batch_size=batch_size, validation_data=(test_X,test_y)) 
        # STORE FINAL WEIGHTS
        self.weights = model.get_weights()

        self.tr_error.append(hist.history)
        elapsed_time = time.time() - start_time
        print('Training completed with ~ {} seconds'.format(round(elapsed_time)))
        self.plot_training()

    def plot_training(self):
        print('Plotting graph...')
        tr_loss = []
        val_loss = []
        for d in self.tr_error:
            tr_loss.append(d['loss'])
            val_loss.append(d['val_loss'])

        # h = range(0, 2400+epoch)
        z = np.array(tr_loss[:]) # TRAINING LOSS
        z1 = np.concatenate(z)
        v1 = np.concatenate(np.array(val_loss[:])) #VALIDATION LOSS

        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        ax.plot(z1, label = 'Training error')
        ax.plot(v1, 'r', label = 'Validation error')
        legend = ax.legend(loc='upper right', shadow=True)

        plt.title('{} init lr = {}'.format(self.mode, self.lr_rate))
        plt.ylabel('Training Error')
        plt.xlabel('nr of epoch') 

        fig.savefig(self.dir_name+'/full_{}_epoch_{}_lr_{}.png'.format(self.mode, self.epochs*5, self.lr_rate))
        if self.SAVE_WTS:
            store_data = {'weights':self.weights, 'tr_error' : z1, 'val_error' : v1}

            scipy.io.savemat(self.dir_name+'/full_{}_init_epoch_{}_lr_{}.mat'.format(self.mode, self.epochs*5,self.lr_rate), store_data)

    def init_fun(self, mode, inp=None, hid=None, x=None, y=None, z=None):
        if mode == 'Identity':
            return eye_init(inp, hid)
        if mode == 'LS':
            return leastsq_init(x, y, z)

    def _create_layer(
        self,
        input_shape, 
        units,
        activation = 'relu'
    ):
        input_layer = Dense(
            units=units,
            input_shape=(input_shape,),
            activation = activation
        )
        return input_layer
        
    def createModel(self, input_shape, units, depth, activation='relu'):
        model = Sequential()
        for i in range(depth): #self.units[1:]:
            layer = self._create_layer(
                input_shape = input_shape,
                units = units,
                activation = activation #self.activations['h']
            )
            model.add(layer)

        return model

    # Sum of square errors:
    def differ(y_true, y_pred):

        return K.mean((y_true - y_pred)**2)
