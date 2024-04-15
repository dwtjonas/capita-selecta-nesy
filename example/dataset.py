import string

import numpy as np
import operator
import random
from nesy.parser import parse_program, parse_clause

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from itertools import product
from torch.utils.data import default_collate

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

LIST_VARS = list(string.ascii_uppercase)
HASYV2 = None
DATASET = None
TARGETS = None
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def filter_data(n):
    characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+']
    #characters = ['0', '1', '2', '3']
    global HASYV2
    HASYV2 = unpickle('HASYv2')
    indices_list = []
    element_list = []
    ctr = 0
    for x in HASYV2['latex_symbol']:
        if x in characters[0:n]:
            indices_list.append(ctr)
            element_list.append(x)
        ctr += 1
    set_target(element_list)
    print(len(indices_list))
    return indices_list

def set_target(list):
    global TARGETS
    res = np.empty(0)
    for i in list:
        if(i == "+"):
            res = np.append(res,10)
        else :
            res = np.append(res,int(i))
    TARGETS = res

def preprocesses(list):
    global DATASET
    selected_data = np.concatenate([HASYV2['data'][..., i:i+1] for i in list], axis=-1)
    selected_data = selected_data[:, :, 0, :]
    transformed_dataset = np.transpose(selected_data, (2, 0, 1))
    DATASET = transformed_dataset


def custom_collate(batch):
    batch = tuple(zip(*batch))
    return default_collate(batch[0]), batch[1], default_collate(batch[2])

operations = ["+","*"]
#For generating something of template "multiplication(X,Y,Z) :- digit(X,N1), digit(Y,N2), multiply(N1,N2,Z).\n"
def generate_operation_string(n, list):
    s = "operation("
    #s += f"Op,"
    for i in range(n+1):
        s +=f"{list[i]},"
    s += "Z) :- "
    return s

def generate_digit_strings_op(n, list):
    s = ""
    for i in range(n+1):
        s += f"digit({list[i]},N{i+1}), "
    return s

def generate_op_strings(n, list):
    s = f"op(Op,N1), "
    return s

def generate_solve_string(n, list):
    s = "add("
    for i in range(1,n+2):
        s += f"N{i},"
    s += "Z).\n"
    return s

#For generating somthing of template solve(+,1,1,2,4)
def generate_math_facts(n_digits, n_classes):
    def augment_numbers(arr, n, current=[], result=""):
        if len(current) == len(arr):
            for op in ["+"]:
                # Calculate the result based on the operation
                if op == "+":
                    result_value = sum(current)
                elif op == "*":
                    result_value = 1
                    for num in current:
                        result_value *= num
                elif op == "-":
                    result_value = current[0]
                    for num in current[1:]:
                        result_value -= num
                # Append the fact to the result string
                result += f"solve({10},{','.join(map(str, current))},{result_value}). "
            return result
        for i in range(n):
            current.append(i)
            result = augment_numbers(arr, n, current, result)
            current.pop()
        return result

    arr = [0] * n_digits
    result = augment_numbers(arr, n_classes)
    return result

def generate_queries_m(n_digits):
    res = "operation("
    for i in range(n_digits+1):
        res += f"tensor(images, {i}), "
    res += "{})."
    return res

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

def generate_queries_ops(n_digits):
    res = "operation("
    for i in range(n_digits):
        res += f"tensor(images, {i}), "
    res += "{})."
    return res

class AdditionTask(Dataset):

    def __init__(self, n=2, train=True, n_classes=4, nr_examples=None):
        #assert n == 2, "Only n=2 is supported at the moment"
        self.train = train
        preprocesses(filter_data(n_classes))
        # We iterate over the MNIST dataset to apply the transform
        self.original_images = []
        self.original_targets = []
        ctr = 0
        mnist = MNIST('data/MNIST', train=train, download=False, transform=transform)
        for x, y in mnist:
            if y < n_classes:
                self.original_images.append(x)
                self.original_targets.append(y)
        '''
        #Used for other dataset
        for x in DATASET:
            self.original_images.append(torch.tensor(x))
        for y in TARGETS:
            self.original_targets.append(y)
        '''
        self.original_images = torch.stack(self.original_images)
        self.original_targets = torch.tensor(self.original_targets)
        self.n_classes = n_classes
        self.num_digits = n
        program_string = ""
        addition_string = generate_addition_string(n, LIST_VARS)
        addition_string += generate_digit_strings(n, LIST_VARS)
        addition_string += generate_add_string(n, LIST_VARS)
        program_string += addition_string
        program_string += generate_add_facts(n, n_classes)
        """
        program_string += "\n".join(
            [f"add({x}, {y}, {x + y})." for x in range(self.n_classes) for y in range(self.n_classes)])
        """
        program_string += "\n"
        program_string += "\n".join(
            [f"nn(digit, tensor(images, {x}), {y}) :: digit(tensor(images, {x}),{y})." for x, y in
             product(range(self.num_digits), range(self.n_classes))])
        # program_string += "\n"
        # program_string += "\n".join(
        #     [f"nn(digit, tensor(images, {x}), {10}) :: digit(tensor(images, {x}),{10})." for x in
        #         range(self.num_digits)])
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
class NumOpsTask(Dataset):

    def __init__(self, n=2, train=True, n_classes=10, nr_examples=None, num_operators=2):
        assert n == 2, "Only n=2 is supported at the moment"
        self.train = train
        self.operator_images = []
        self.operator_targets = []

        # We iterate over the MNIST dataset to apply the transform
        self.original_images = []
        self.original_targets = []
        for x,y in  MNIST('data/MNIST/', train=train, download=True, transform=transform):
            if y < n_classes:
                self.original_images.append(x)
                self.original_targets.append(y)
            if y < num_operators:
                self.operator_images.append(x)
                self.operator_targets.append(y)

        self.original_images = torch.stack(self.original_images)
        self.original_targets = torch.tensor(self.original_targets)
        self.n_classes = n_classes
        self.num_digits = n
        self.len_operator = len(self.operator_images)
        program_string = "operation(Op,X,Y,Z) :- operator(Op,N1), digit(X,N2), digit(Y,N3), op(N1,N2,N3,Z).\n"
        program_string += "\n".join(
            [f"op(1,{x}, {y}, {x * y})." for x in range(self.n_classes) for y in range(self.n_classes)])
        program_string += "\n".join(
            [f"op(0,{x}, {y}, {x + y})." for x in range(self.n_classes) for y in range(self.n_classes)])
        program_string += "\n"
        program_string += "\n".join(
            [f"nn(digit, tensor(images, {x}), {y}) :: digit(tensor(images, {x}),{y})." for x, y in
             product(range(1, self.num_digits + 1), range(self.n_classes))])
        program_string += "\n".join(
            [f"nn(operator, tensor(images, 0), {y}) :: operator(tensor(images, 0),{y})." for  y in
             range(num_operators)])
        self.program = parse_program(program_string)

        if nr_examples is not None:
            if nr_examples > self.nr_examples:
                raise ValueError('nr_examples exceeds to number of available examples in this dataset')
            else:
                self.nr_examples = nr_examples
        else:
            self.nr_examples = len(self.original_images) // self.num_digits

    def __getitem__(self, index):
        images = self.original_images[index * self.num_digits: (index + 1) * self.num_digits ]
        targets = self.original_targets[index * self.num_digits: (index + 1) * self.num_digits]
        random_index = random.randint(0, self.len_operator-1)
        operator_image = self.operator_images[random_index]
        operator_target = self.operator_targets[random_index]

        symbol = operator_target # 0 = add, 1 = mult
        if symbol == torch.tensor(1):
            target = targets[0]
            for t in targets[1:]:
                target *= t
        else:
            target = sum(targets)

        if self.train:
            # In MNIST Addition, training queries for a single pair of images check for a given sum (i.e. the target)
            # Therefore, we have a List[Term], each element of the list correspond to a single pair of images

            query = parse_program(generate_queries_ops(self.num_digits + 1).format(target))[0].term
            tensor_sources = {"images": images, "operator": operator_image}
            return tensor_sources, query, torch.tensor([1.0])
        else:
            # In MNIST Addition, testing queries for a single pair of images check for all possible sums.
            # In this way, we can compute the most probable sum.
            # Therefore, we have a List[List[Term]], each element of the outer list correspond to a single pair of
            # images. Each element of the inner list correspond to a possible sum.

            queries = [parse_program(generate_queries_ops(self.num_digits + 1).format(target))[0].term
                       for z in range(self.n_classes * 2 - 1)]
            tensor_sources = {"images": images, "operator": operator_image}
            return tensor_sources, queries, target

    def dataloader(self, batch_size=2, shuffle=None, num_workers=0):
        if shuffle is None:
            shuffle = self.train

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate,
                          num_workers=num_workers)

    def __len__(self):
        return self.nr_examples
