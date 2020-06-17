import numpy as np
import time
import tensorflow as tf
from tensorflow              import keras
from tensorflow.keras        import layers, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
tf.keras.backend.clear_session()

basePath= 'sd_GSCmdV2/'
fn= 'gscV2_data.npz'


t0= time.time()

z= np.load(basePath+fn)

x_train=    z['x_trainWithSil']    
y_train=    z['y_trainWithSil']    
x_val=      z['x_val']      
y_val=      z['y_val']
x_test=     z['x_test']     
y_test=     z['y_test']     

fnModel= 'ryModel_2.hdf5'

print("Datos de ({}), se guardara el modelo en {}".format(fn, fnModel))


#características que se van a extraer de los datos originales
def ryFeature(x, 
           sample_rate= 16000, 
           
           frame_length= 1024,
           frame_step=    128,  # frame_length//2
           
           num_mel_bins=     128,
           lower_edge_hertz= 20,     # 0
           upper_edge_hertz= 16000/2, # sample_rate/2   
           
           mfcc_dim= 13
           ):
    
    stfts= tf.signal.stft(x, 
                          frame_length, #=  256, #1024, 
                          frame_step, #=    128,
                          #fft_length= 1024
                          pad_end=True
                          )
    
    spectrograms=     tf.abs(stfts)
    log_spectrograms= tf.math.log(spectrograms + 1e-10)
    
    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins= stfts.shape[-1]  #.value
    
    linear_to_mel_weight_matrix= tf.signal.linear_to_mel_weight_matrix(
          num_mel_bins, 
          num_spectrogram_bins, 
          sample_rate, 
          lower_edge_hertz,
          upper_edge_hertz)
    
    mel_spectrograms= tf.tensordot(
          spectrograms, 
          linear_to_mel_weight_matrix, 1)
    
    mel_spectrograms.set_shape(
          spectrograms.shape[:-1].concatenate(
              linear_to_mel_weight_matrix.shape[-1:]))
    
    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms= tf.math.log(mel_spectrograms + 1e-10)
    
    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfccs= tf.signal.mfccs_from_log_mel_spectrograms(
          log_mel_spectrograms)[..., :mfcc_dim]
    
    feature= {'mfcc':               mfccs, 
              'log_mel_spectrogram':log_mel_spectrograms, 
              'log_spectrogram':    log_spectrograms, 
              'spectrogram':        spectrograms}
    
    return  feature




#de todas las caracteristicas que se extraen, de decide cuales dejar.
def get_all_fearure(all_x, batch_size= 1000):
    t0= time.time()
    
    x= all_x.astype(np.float32)
    
    
    i=0
    XL=[]
    while i < x.shape[0]:
        
        if i+batch_size<=x.shape[0]:
            xx= x[i:i+batch_size]
        else:
            xx= x[i:]
        
        XX= ryFeature(xx)
        X= XX['log_mel_spectrogram'] #se puedenw intentar con otras 
        
        X= X.numpy().astype(np.float32)
        
        i  += batch_size
        XL += [X]
    
    XL= np.concatenate(XL)
    print('XL.shape={}'.format(XL.shape))
    
    dt= time.time()-t0
    print('tf.signal.stft, dt= {}'.format(dt))

    return XL


#para normalizar datos
def normalize(x, axis= None):   
    if axis== None:
        x= (x-tf.math.reduce_mean(x))/tf.math.reduce_std(x)
    else:
        x= (x-tf.math.reduce_mean(x, axis= axis))/tf.math.reduce_std(x, axis= axis)
    
    return x




#se extraen las caracterñisticas deseadas
X_test=     get_all_fearure(x_test)
X_val=      get_all_fearure(x_val)
X_train=    get_all_fearure(x_train)

#t0= time.time()
dt= time.time()- t0
print('Se extrayeron las caracteristicas en {:.3f}'.format(dt))

nTime, nFreq= X_train[0].shape


#se normalizan los datos

X_train= X_train.reshape(-1, nTime, nFreq).astype('float32') 
X_val=   X_val.reshape(-1, nTime, nFreq).astype('float32') 
X_test=  X_test.reshape( -1, nTime, nFreq).astype('float32') 
#X_testREAL=  X_testREAL.reshape( -1, nTime, nFreq, 1).astype('float32') 

X_train=     normalize(X_train)#, axis=0)  # normalized for the all set, many utterence
X_val=       normalize(X_val)#, axis=0)
X_test=      normalize(X_test)#, axis=0)
#X_testREAL=  normalize(X_testREAL)#, axis=0)


#numero de categorias a entrenar
nCategs= len(set(y_train))




#modelo
x= Input(shape= (nTime, nFreq))

h= x

h= Conv1D(40, 101, strides=8, activation='relu', padding='same')(h)
h= tf.math.log(tf.abs(h) + 1)
h = normalize(h)
h= AveragePooling1D(2, padding='same')(h)

h= Conv1D(160, 25, strides=2, activation='relu', padding='same')(h)
h = normalize(h)
h= AveragePooling1D(2, padding='same')(h)
h= Dropout(0.1)(h)

h= Conv1D(160, 9, strides=1, activation='relu', padding='same')(h)
h = normalize(h)
h= AveragePooling1D(2, padding='same')(h)
h= Dropout(0.1)(h)

h= Conv1D(160,  9, strides=1, activation='relu', padding='same')(h)
h = normalize(h)
h= AveragePooling1D(2, padding='same')(h)
h= Dropout(0.1)(h)

h= Conv1D(160,  9, strides=1, activation='relu', padding='same')(h)
h = normalize(h)
h= AveragePooling1D(2, padding='same')(h)
h= Dropout(0.1)(h)

h= Conv1D(160, 9, strides=1, activation='relu', padding='same')(h)
h = normalize(h)
h= AveragePooling1D(2, padding='same')(h)
h= Dropout(0.1)(h)



h= Flatten()(h)


h= Dense(nCategs,  activation='softmax')(h)

y= h

m= Model(inputs=  x, 
         outputs= y)

m.summary()




m.compile(  
        loss=    'sparse_categorical_crossentropy',
        metrics= ['accuracy'])


es= EarlyStopping(
        monitor=   'val_loss', 
        min_delta= 1e-10,
        patience=  10, 
        mode=      'min', 
        verbose=   1) 



mc= ModelCheckpoint(fnModel, 
        monitor=    'val_accuracy', 
        verbose=    1, 
        save_best_only= True, 
        mode=      'max')

t0= time.time()

h= m.fit(X_train, y_train,
         
        batch_size=500, #1000, # 1000
        epochs=    200,
        
        callbacks=[mc],
        
        #validation_split= 0.1
        validation_data= (X_val, y_val)
        )



tf.keras.backend.clear_session()
fnModel= 'ryModel_3.hdf5'


#se normalizan los datos

X_train= X_train.reshape(-1, nTime, nFreq,1).astype('float32') 
X_val=   X_val.reshape(-1, nTime, nFreq, 1).astype('float32') 
X_test=  X_test.reshape( -1, nTime, nFreq, 1).astype('float32') 
#X_testREAL=  X_testREAL.reshape( -1, nTime, nFreq, 1).astype('float32') 

X_train=     normalize(X_train)#, axis=0)  # normalized for the all set, many utterence
X_val=       normalize(X_val)#, axis=0)
X_test=      normalize(X_test)#, axis=0)
#X_testREAL=  normalize(X_testREAL)#, axis=0)


#numero de categorias a entrenar
nCategs= len(set(y_train))




x= Input(shape= (nTime, nFreq, 1))

h= x


h= Conv2D(8,   (16,16), activation='relu', padding='same')(h)
h= MaxPooling2D((4,4), padding='same')(h)
h= Dropout(0.2)(h)

h= Conv2D(16,   (8,8), activation='relu', padding='same')(h)
h= MaxPooling2D((4,4), padding='same')(h)
h= Dropout(0.2)(h)

h= Flatten()(h)

h= Dense(256,  activation='relu')(h)
h= Dropout(0.2)(h)


h= Dense(nCategs,  activation='softmax')(h)

y= h

m1= Model(inputs=  x, 
         outputs= y)

m1.summary()



m1.compile(  
        loss=    'sparse_categorical_crossentropy',
        metrics= ['accuracy'])


es= EarlyStopping(
        monitor=   'val_loss', 
        min_delta= 1e-10,
        patience=  10, 
        mode=      'min', 
        verbose=   1) 



mc= ModelCheckpoint(fnModel, 
        monitor=    'val_accuracy', 
        verbose=    1, 
        save_best_only= True, 
        mode=      'max')

t0= time.time()

h= m1.fit(X_train, y_train,
         
        batch_size=500, #1000, # 1000
        epochs=    200,
        
        callbacks=[mc],
        
        #validation_split= 0.1
        validation_data= (X_val, y_val)
        )
