import string

from nesy.parser import parse_program, parse_clause

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from itertools import product
from torch.utils.data import default_collate

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

LIST_VARS = list(string.ascii_uppercase)

def custom_collate(batch):
    batch = tuple(zip(*batch))
    return default_collate(batch[0]), batch[1], default_collate(batch[2])

#For generating something of template "addition(X,Y,Z) :- digit(X,N1), digit(Y,N2), add(N1,N2,Z).\n"
def generate_addition_string(n, list):
    s = "addition("
    for i in range(n):
        s +=f"{list[i]},"
    s += "Z) :- "
    return s

def generate_digit_strings(n, list):
    s = ""
    for i in range(n):
        s += f"digit({list[i]},N{i+1}), "
    return s

def generate_add_string(n, list):
    s = "add("
    for i in range(1,n+1):
        s += f"N{i},"
    s += "Z).\n"
    return s

#For generating somthing of template add(1,1,2,4)
def generate_add_facts(n_digits, n_classes):
    def augment_numbers(arr, n, current=[], result=""):
        if len(current) == len(arr):
            current_sum = sum(current)
            result += f"add{tuple(current + [current_sum])}.".replace(" ", "")
            result += " "
            return result
        for i in range(n):
            current.append(i)
            result = augment_numbers(arr, n, current, result)
            current.pop()
        return result
    arr = [0] * n_digits
    return augment_numbers(arr, n_classes)

def generate_queries(n_digits):
    res = "addition("
    for i in range(n_digits):
        res += f"tensor(images, {i}), "
    res += "{})."
    return res

class AdditionTask(Dataset):

    def __init__(self, n=2, train=True, n_classes=4, nr_examples=None):
        #assert n == 2, "Only n=2 is supported at the moment"
        self.train = train

        # We iterate over the MNIST dataset to apply the transform
        self.original_images = []
        self.original_targets = []
        for x,y in  MNIST('data/MNIST/', train=train, download=True, transform=transform):
            if y < n_classes:
                self.original_images.append(x)
                self.original_targets.append(y)
        self.original_images = torch.stack(self.original_images)
        self.original_targets = torch.tensor(self.original_targets)
        self.n_classes = n_classes
        self.num_digits = n
        program_string = ""
        addition_string = generate_addition_string(n,LIST_VARS)
        addition_string +=generate_digit_strings(n,LIST_VARS)
        addition_string +=generate_add_string(n,LIST_VARS)
        program_string += addition_string
        program_string += generate_add_facts(n, n_classes)
        """
        program_string += "\n".join(
            [f"add({x}, {y}, {x + y})." for x in range(self.n_classes) for y in range(self.n_classes)])
        """
        program_string += "\n"
        #print("\n".join(
        #    [f"nn(digit, tensor(images, {x}), {y}) :: digit(tensor(images, {x}),{y})." for x, y in
        #     product(range(self.num_digits), range(self.n_classes))]))
        program_string += "\n".join(
            [f"nn(digit, tensor(images, {x}), {y}) :: digit(tensor(images, {x}),{y})." for x, y in
             product(range(self.num_digits), range(self.n_classes))])
        self.program = parse_program(program_string)

        if nr_examples is not None:
            if nr_examples > self.nr_examples:
                raise ValueError('nr_examples exceeds to number of available examples in this dataset')
            else:
                self.nr_examples = nr_examples
        else:
            self.nr_examples = len(self.original_images) // self.num_digits

    def __getitem__(self, index):
        images = self.original_images[index * self.num_digits: (index + 1) * self.num_digits]
        targets = self.original_targets[index * self.num_digits: (index + 1) * self.num_digits]
        target = int(targets.sum())

        if self.train:
            # In MNIST Addition, training queries for a single pair of images check for a given sum (i.e. the target)
            # Therefore, we have a List[Term], each element of the list correspond to a single pair of images

            query = parse_program(generate_queries(self.num_digits).format(target))[0].term
            tensor_sources = {"images": images}
            return tensor_sources, query, torch.tensor([1.0])
        else:
            # In MNIST Addition, testing queries for a single pair of images check for all possible sums.
            # In this way, we can compute the most probable sum.
            # Therefore, we have a List[List[Term]], each element of the outer list correspond to a single pair of
            # images. Each element of the inner list correspond to a possible sum.

            queries = [parse_program(generate_queries(self.num_digits).format(z))[0].term
                       for z in range(self.n_classes * 2 - 1)]
            tensor_sources = {"images": images}
            return tensor_sources, queries, target

    def dataloader(self, batch_size=2, shuffle=None, num_workers=0):
        if shuffle is None:
            shuffle = self.train

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate,
                          num_workers=num_workers)

    def __len__(self):
        return self.nr_examples
