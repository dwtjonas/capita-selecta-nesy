from nesy.model import NeSyModel, MNISTEncoder
from dataset import AdditionTask
from nesy.logic import ForwardChaining
from nesy.semantics import GodelTNorm, LukasieviczTNorm, ProductTNorm, SumProductSemiring

import torch
import pytorch_lightning as pl

n_classes = 2
NUMBER_OF_DIGITS = 2
num_workers = 0
batch_size = 2

task_train = AdditionTask(n=NUMBER_OF_DIGITS, n_classes=n_classes)
task_test = AdditionTask(n=NUMBER_OF_DIGITS, n_classes=n_classes, train=False)

neural_predicates = torch.nn.ModuleDict({"digit": MNISTEncoder(task_train.n_classes)})

model = NeSyModel(program=task_train.program,
                logic_engine=ForwardChaining(),
                neural_predicates=neural_predicates,
                label_semantics=ProductTNorm())

trainer = pl.Trainer(max_epochs=1)
trainer.fit(model=model,
            train_dataloaders=task_train.dataloader(batch_size=batch_size, num_workers=num_workers),
            val_dataloaders=task_test.dataloader(batch_size=batch_size, num_workers=num_workers))
