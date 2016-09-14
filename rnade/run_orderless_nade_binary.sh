export DATASETSPATH=.
export RESULTSPATH=./output
export PYTHONPATH=./buml:$PYTHONPATH
mkdir -p output
echo $DATASETSPATH
echo $RESULTSPATH
python orderlessNADE.py --theano --form Bernoulli --dataset binarized_mnist.hdf5 --training_route /train --validation_route /validation --test_route /test --samples_name data --hlayers 2  --layerwise --lr 0.001 --epoch_size 100 --momentum 0.9 --units 500  --pretraining_epochs 1 --validation_loops 16 --epochs 200 --batch_size 100 --show_training_stop mnist