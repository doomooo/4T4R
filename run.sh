#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:/modules

# train the order of data
cd train_order
python3 ModelTrainVQ.py &
cd ..

# Train data with sort with common quant method
cd pytorch_template
python3 ModelTrainSplitStartNoQuant.py

# Finetune data with 2bit 3bit mixed quant
python3 ModelFineTune_Best_bit25.py
cd ..


# Train data with predicted sort
cd combined_model
python3 ModelFinetune_IndexVQ_25.py
cp model_finetune/* ../submit_pytorch/modelSubmit
cd ..


mkdir /data/prediction_result
cp -r submit_pytorch /data/prediction_result
