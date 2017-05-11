import tensorflow as tf
import deepchem as dc
import numpy as np
graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()
loader = dc.data.data_loader.CSVLoader( tasks=['LogS'], smiles_field="CANONICAL_SMILES", id_field="CMPD_CHEMBLID", featurizer=graph_featurizer )
dataset = loader.featurize( './bioactivity.csv' )

splitter = dc.splits.splitters.RandomSplitter()
trainset,testset = splitter.train_test_split( dataset )

hp = dc.molnet.preset_hyper_parameters
param = hp.hps[ 'graphconvreg' ]
print(param['batch_size'])
g = tf.Graph()
graph_model = dc.nn.SequentialGraph( 75 )
graph_model.add( dc.nn.GraphConv( int(param['n_filters']), 75, activation='relu' ))
graph_model.add( dc.nn.BatchNormalization( epsilon=1e-5, mode=1 ))
graph_model.add( dc.nn.GraphPool() )
graph_model.add( dc.nn.GraphConv( int(param['n_filters']), int(param['n_filters']), activation='relu' ))
graph_model.add( dc.nn.BatchNormalization( epsilon=1e-5, mode=1 ))
graph_model.add( dc.nn.GraphPool() )
graph_model.add( dc.nn.Dense( int(param['n_fully_connected_nodes']), int(param['n_filters']), activation='relu' ))
graph_model.add( dc.nn.BatchNormalization( epsilon=1e-5, mode=1 ))
#graph_model.add( dc.nn.GraphGather(param['batch_size'], activation='tanh'))
graph_model.add( dc.nn.GraphGather( 10 , activation='tanh'))

with tf.Session() as sess:
    model_graphconv = dc.models.MultitaskGraphRegressor( graph_model,
                                                      1,
                                                      75,
                                                     batch_size=10,
                                                     learning_rate = param['learning_rate'],
                                                     optimizer_type = 'adam',
                                                     beta1=.9,beta2=.999)
    model_graphconv.fit( trainset, nb_epoch=30 )

train_scores = {}
regression_metric = dc.metrics.Metric( dc.metrics.pearson_r2_score, np.mean )
train_scores['graphconvreg'] = model_graphconv.evaluate( trainset,[ regression_metric ]  )
p=model_graphconv.predict( testset )
'''
for i in range( len(p )):
    print( p[i], testset.y[i] )
'''
print(train_scores) 
