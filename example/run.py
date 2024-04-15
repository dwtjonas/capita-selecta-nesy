from nesy.model import NeSyModel, MNISTEncoder
from dataset import AdditionTask, NumOpsTask
from nesy.logic import ForwardChaining
from nesy.semantics import GodelTNorm, LukasieviczTNorm, ProductTNorm, SumProductSemiring

import torch
import pytorch_lightning as pl

n_classes = 2
number_digits = 2
batch_size = 10
max_epochs = 5
task_train = NumOpsTask(n_classes=n_classes, n=number_digits)
task_test = NumOpsTask(n_classes=n_classes, train=False, n = number_digits)

neural_predicates = torch.nn.ModuleDict({"digit": MNISTEncoder(task_train.n_classes)})

model = NeSyModel(program=task_train.program,
                logic_engine=ForwardChaining(),
                neural_predicates=neural_predicates,
                label_semantics=(ProductTNorm()))

trainer = pl.Trainer(max_epochs=max_epochs)
trainer.fit(model=model,
            train_dataloaders=task_train.dataloader(batch_size=batch_size),
            val_dataloaders=task_test.dataloader(batch_size=batch_size))
print("run ended\n")

