import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

filename = 'alice.txt'
raw_txt = open( filename ).read()
raw_txt = raw_txt.lower()

chars = sorted( list( set( raw_txt )))
char_to_int = dict((c, i) for i, c in enumerate( chars ))

n_chars = len( raw_txt )
n_vocab = len( chars )

seq_len = 100
dataX = []
dataY = []

for i in range( 0, n_chars - seq_len, 1 ):
    seq_in = raw_txt[ i:i+seq_len ]
    seq_out = raw_txt[ i + seq_len ]
    dataX.append( [ char_to_int[ char ] for char in seq_in] )
    dataY.append( char_to_int[ seq_out ] )
n_patterns = len( dataX )
#print( n_patterns )
X = np.reshape( dataX, ( n_patterns, seq_len, 1 ))
#print(X)
X = X / float( n_vocab )
y = np_utils.to_categorical( dataY )

model = Sequential()
model.add( LSTM( 256, input_shape=( X.shape[1], X.shape[2] )))
model.add( Dropout( 0.2 ))
model.add( Dense( y.shape[1], activation = 'softmax' ))
#model.compile( loss='categorical_crossentropy', optimizer = 'adam' )

filepath = 'weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint( filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
callbacks_list = [ checkpoint ]
#model.fit( X, y, epochs=20, batch_size=128, callbacks=callbacks_list )

fname = 'weights-improvement-19-2.0503.hdf5'
model.load_weights( fname )
model.compile( loss='categorical_crossentropy', optimizer='adam' )

int_to_char = dict( (i,c) for i,c in enumerate( chars ) )
print( chars ) 
start = np.random.randint( 0, len( dataX ) - 1 )
pattern = dataX[ start ]
print( pattern )
print( 'Seed:' )
print( '\'' , ''.join( [ int_to_char[ val ] for val in pattern ] ), '\'')
print( len( pattern) )
import sys
for i in range( 10000 ):
    x = np.reshape( pattern, ( 1, len( pattern ), 1 ))
    x = x/float( n_vocab )
    prediction = model.predict( x, verbose = 0 )
    index = np.argmax( prediction )
    res = int_to_char[ index ]
    seq_in = [ int_to_char[ val ] for val in pattern ]
    sys.stdout.write( res )
    pattern.append( index )
    pattern = pattern[ 1:len( pattern ) ]
print( "done ;-)" )
