import numpy as np
import pandas as pd
import sys
import argparse
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import Draw

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
        if act >= 6:
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
    embed.sprite.image_path = os.path.join( FLAGS.data_dir + '/mols.png' )
    embed.sprite.single_image_dim.extend( [100, 100] )
    projector.visualize_embeddings( writer, config )
    saver.save( sess, os.path.join(FLAGS.log_dir, 'projector/amodel.ckpt'), global_step=len(X) )
def generate_metadata_file():
    sdf = Chem.SDMolSupplier( FLAGS.sdf )
    Y = getResponse( [ mol for mol in sdf ])
    def save_metadata( file ):
        with open( file, 'w' ) as f:
            f.write('id\tactivity_class\n')
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
    parser.add_argument( '--log_dir', type=str, default='/Users/iwatobipen/develop/py35env/testfolder/tensorflowtest/mollog' )
    parser.add_argument( '--data_dir', type=str, default='/Users/iwatobipen/develop/py35env/testfolder/tensorflowtest/mollog')

    FLAGS, unparsed = parser.parse_known_args()
    sdf = [ mol for mol in Chem.SDMolSupplier( FLAGS.sdf ) ]
    im = Draw.MolsToGridImage( sdf, molsPerRow=10, subImgSize=( 100, 100 ))
    im.save( os.path.join( FLAGS.data_dir + '/mols.png' ))
 
    
    tf.app.run( main=main, argv=[sys.argv[0]] + unparsed )


