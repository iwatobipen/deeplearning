import numpy as np
import pandas as pd
import sys
import argparse
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = None

def getFpArr( mols, nBits = 1024 ):
    fps = [ AllChem.GetMorganFingerprintAsBitVect( mol, 2, nBits=nBits ) for mol in mols ]
    X = []
    for fp in fps:
        arr = np.zeros( (1,) )
        DataStructs.ConvertToNumpyArray( fp, arr )
        X.append( arr )
    return np.array( X )

def getResponse( mols, prop="ACTIVITY" ):
    Y = []
    for mol in mols:
        act = mol.GetProp( prop )
        act = 9. - np.log10( float( act ) )
        if act >= 7:
            Y.append(np.asarray( [1,0] ))
        else:
            Y.append(np.asarray( [0,1] ))
    return np.asarray( Y )


def generate_embeddings():
    sdf = Chem.SDMolSupplier( FLAGS.sdf )
    X = getFpArr( [ mol for mol in sdf ]  )
    sess = tf.InteractiveSession()
    with tf.device( '/cpu:0' ):
        embedding = tf.Variable( tf.stack( X[:], axis=0 ), trainable=False, name='embedding' )
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter( FLAGS.log_dir+'/projector', sess.graph )
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'embedding:0'
    embed.metadata_path = os.path.join( FLAGS.log_dir + '/projector/metadata.tsv' )
    projector.visualize_embeddings( writer, config )
    saver.save( sess, os.path.join(FLAGS.log_dir, 'projector/amodel.ckpt'), global_step=len(X) )
def generate_metadata_file():
    sdf = Chem.SDMolSupplier( FLAGS.sdf )
    Y = getResponse( [ mol for mol in sdf ])
    def save_metadata( file ):
        with open( file, 'w' ) as f:
            for i in range( Y.shape[0] ):
                c = np.nonzero( Y[i] )[0][0]
                f.write( '{}\t{}\n'.format( i, c ))
    save_metadata( FLAGS.log_dir + '/projector/metadata.tsv' )

def main(_):
    if tf.gfile.Exists( FLAGS.log_dir+'/projector' ):
        tf.gfile.DeleteRecursively( FLAGS.log_dir+'/projector' )
        tf.gfile.MkDir( FLAGS.log_dir + '/projector' )
    tf.gfile.MakeDirs( FLAGS.log_dir + '/projector' )
    generate_metadata_file()
    generate_embeddings()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--sdf', type=str )
    parser.add_argument( '--log_dir', type=str, default='Fullpat/mollog' )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run( main=main, argv=[sys.argv[0]] + unparsed )




if __name__ == '__main__':
    filename = sys.argv[1]
    sdf = [ mol for mol in Chem.SDMolSupplier( filename ) ]
    X = np.asarray(  getFpArr( sdf ))
    #X = np.reshape( X, ( X.shape[0], 1, X.shape[1]))
    Y = getResponse( sdf )

    trainx, testx, trainy, testy = train_test_split( X, Y, test_size=0.2, random_state=0 )
    trainx, testx, trainy, testy = np.asarray( trainx ), np.asarray( testx ), np.asarray( trainy ), np.asarray( testy )
    model = base_model()
    """
    estimator = KerasRegressor( build_fn = base_model,
                                nb_epoch=100,
                                batch_size=50,
                                 )
    estimator.fit( trainx, trainy )
    """
    model.fit( trainx, trainy, validation_data=(testx,testy), callbacks=[tb_cb] )
    pred_y = model.predict( testx )
    r2 = r2_score( testy, pred_y )
    rmse = mean_squared_error( testy, pred_y )
    print( "KERAS: R2 : {0:f}, RMSE : {1:f}".format( r2, rmse ) )
