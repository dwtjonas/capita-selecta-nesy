from nesy.model import NeSyModel, MNISTEncoder
from example.dataset import AdditionTask
from nesy.logic import ForwardChaining
from nesy.semantics import GodelTNorm, LukasieviczTNorm, SumProductSemiring

import torch
import pytorch_lightning as pl

for i in range(4,5 ):
    print(i)
    print('\n')
    task_train = AdditionTask(n_classes=i)
    task_test = AdditionTask(n_classes=i, train=False)

    neural_predicates = torch.nn.ModuleDict({"digit": MNISTEncoder(task_train.n_classes)})

    model = NeSyModel(program=task_train.program,
                    logic_engine=ForwardChaining(),
                    neural_predicates=neural_predicates,
                    label_semantics=GodelTNorm())

    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model=model,
                train_dataloaders=task_train.dataloader(batch_size=3),
                val_dataloaders=task_test.dataloader(batch_size=3))
    print("run ended\n")
