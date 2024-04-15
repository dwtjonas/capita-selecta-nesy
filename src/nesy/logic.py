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
        #print("\n")
        #print(program)
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

        # Derrive new facts using by substituing query into clause
        new = []
        for c in clause.body:
            new_args = c.arguments
            for i, arg in enumerate(clause.head.arguments):
                replace_with = query.arguments[i]
                new_args = tuple(replace_with if x == arg else x for x in new_args)
            new.append(Term(c.functor, new_args))

        add_clause = new[-1]
        other_clauses = new[:-1]

        # Find the possible assignments for the free variables
        free_var = tuple(add_clause.arguments[0:-1])
        result = add_clause.arguments[-1]
        free_var_assignments = []
        for f in facts:
            if f.term.functor == 'add' and f.term.arguments[-1] == result:
                free_var_assignments.append(tuple(f.term.arguments[0:-1]))
            if f.term.functor == 'op' and f.term.arguments[-1] == result:
                free_var_assignments.append(tuple(f.term.arguments[0:-1]))           
        
        # Substitute the free variables in all possbile ways
        new_facts = [[] for _ in range(len(free_var_assignments))]
        for i, assignment in enumerate(free_var_assignments):
            for clause in new:
                args = list(clause.arguments)
                # Assign to the free variables
                for j, arg in enumerate(args):
                    for k, var in enumerate(free_var):
                        if arg == var:
                            args[j] = assignment[k]
                new_facts[i].append(Term(clause.functor, tuple(args)))

        # Replace with neural predicates
        for i, g in enumerate(new_facts):
            for j, c in enumerate(g):
                if not c.functor == 'digit' and not c.functor == 'operator': continue
                for f in facts:
                    if f.term == c:
                        new_facts[i][j] = f.weight
                        break
        return new_facts



    