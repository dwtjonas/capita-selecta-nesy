import torch
class Evaluator():

    def __init__(self, label_semantics, neural_predicates):
        self.neural_predicates = neural_predicates
        self.label_semantics = label_semantics

    def evaluate(self, tensor_sources, and_or_tree, queries):
        result = None

        # Test
        if isinstance(queries[0], list):
            result = [[] for _ in range(len(queries))]
            for i, q in enumerate(queries):
                for j in range(len(q)):
                    result[i].append(self.eval_tree(tensor_sources, and_or_tree[i][j], i))
                result[i] = torch.tensor(result[i])

        # Train
        else:
            result = []
            for i in range(len(queries)):
                result.append(self.eval_tree(tensor_sources, and_or_tree[i], i))

        return torch.stack(result)

    
    def eval_tree(self, tensor_sources, tree, batch):     
        for i, b in enumerate(tree): # OR
            for j, c in enumerate(b): # AND
                if c.functor == 'nn':
                    image = int(c.arguments[1].arguments[1].functor)
                    result = int(c.arguments[2].functor)
                    entry = c.arguments[0].functor
                    if entry == 'digit':
                        if(len(tensor_sources) == 2): image -= 1
                        p_digit = self.neural_predicates["digit"](tensor_sources['images'][:,image])[:,result][batch]
                    else:
                        p_digit = self.neural_predicates["digit"](tensor_sources['operator'])[:,result][batch]
                    tree[i][j] = p_digit
                elif c.functor == 'add':
                    tree[i][j] = 1
                elif c.functor == 'op':
                    tree[i][j] = 1
                else:
                    Exception # should not get here
            tree[i] = self.label_semantics.conjunction(tree[i])
        return self.label_semantics.disjunction(tree)
