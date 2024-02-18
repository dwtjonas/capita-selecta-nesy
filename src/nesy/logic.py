from nesy.term import Fact, Term, Clause, Variable
from nesy.parser import parse_term, parse_program
from abc import ABC, abstractmethod
from collections import namedtuple

class LogicEngine(ABC):

    @abstractmethod
    def reason(self, program: list[Clause], queries: list[Term]):
        pass

class LogicNode:
    def __init__(self, value):
        self.value = value
    def evaluate(self):
        pass
class LeafNode(LogicNode):

    def evaluate(self):
        return self.value

class AndNode(LogicNode):

    def evaluate(self):
        return self.value

class OrNode(LogicNode):
    def evaluate(self):
        return self.value

class ForwardChaining(LogicEngine):
    def reason(self, program, queries):
        #query = parse_term("addition(tensor(images,0), tensor(images,1), 2)")
        #print(queries)
        result = None

        # Test
        if isinstance(queries[0], list):
            result = [[] for _ in range(len(queries))]
            for i, q in enumerate(queries):
                for qq in q:
                    result[i].append(self.solve_query(program, qq))

        # Train
        else:
            result = []
            for q in queries:
                result.append(self.solve_query(program, q))

        #print(result)
        return result
    
    def solve_query(self, program, query):
        clause = program[0]
        facts = program[1:]
        new = []
        if clause.head.functor == query.functor:
            for c in clause.body:
                new_args = c.arguments
                for i, arg in enumerate(clause.head.arguments):
                    replace_with = query.arguments[i]
                    new_args = tuple(replace_with if x == arg else x for x in new_args)
                new.append(Term(c.functor, new_args))

        add_clause = None
        other_clauses = []
        for c in new:
            if c.functor == 'add':
                add_clause = c
            else:
                other_clauses.append(c)
        add_facts = []
        for f in facts:
            if f.term.functor == 'add':
                add_facts.append(f)

        free_var = tuple(add_clause.arguments[0:2])
        result = add_clause.arguments[2]
        free_var_assignments = []
        for f in add_facts:
            if f.term.arguments[2] == result:
                free_var_assignments.append(tuple(f.term.arguments[0:2]))
        
        # Substitute
        new_facts = [[] for _ in range(len(free_var_assignments))]
        for i in range(len(free_var_assignments)):
            assign = free_var_assignments[i]
            for c in new:
                new_args = list(c.arguments)
                for j, arg in enumerate(new_args):
                    if arg == free_var[0]:
                        new_args[j] = assign[0]
                    elif arg == free_var[1]:
                        new_args[j] = assign[1]
                new_facts[i].append(Term(c.functor, tuple(new_args)))

        # Replace with neural predicates
        for i, g in enumerate(new_facts):
            for j, c in enumerate(g):
                if c.functor != 'digit': break
                for f in facts:
                    if f.term == c:
                        new_facts[i][j] = f.weight
        return new_facts



    