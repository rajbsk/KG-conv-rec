#!/data/qibin/anaconda3/envs/alchemy/bin/python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Train a model using parlai's standard training loop.

For documentation, see parlai.scripts.train_model.
"""

from parlai.scripts.train_model import TrainLoop, setup_args

if __name__ == "__main__":
    parser = setup_args()
    parser.set_defaults(
        task="redial",
        dict_tokenizer="split",
        model="kbrd",
        dict_file="saved/tmp",
        model_file="saved/kbrd",
        fp16=True,
        batchsize=32,
        n_entity=101158,
        n_relation=410,
        datapath="data_3/",
        dim = 300,
        num_epochs=0,
        validation_patience = 5,
        learningrate=0.001,
        # validation_metric="recall@50",
        validation_metric="base_loss",
        validation_metric_mode='min',
        validation_every_n_secs=60,
        tensorboard_log=True,
        tensorboard_tag="task,model,batchsize,dim,learningrate,model_file",
        tensorboard_metrics="loss,base_loss,kge_loss,l2_loss,acc,auc,recall@1,recall@10,recall@50,recall@100",
    )
    opt = parser.parse_args()
    TrainLoop(opt).train()
