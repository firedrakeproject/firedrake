from enum import Enum
from graphviz import Digraph
from re import search

class GraphType(Enum):
    """Type of graph to create

    PARENT_CHILD: Graph is composed of parent-child relationships only and each node is labelled with all information about the node (its type and all its slots including 'children').
    DAG: DAG visualisation. Each node label is shortened for readability.
    GRAPH: The bits of information about each node (i.e. its slots) are unique nodes in the graph. Note that the visualisation from the this case shows the expression graph as a graph rather than as the dag it truely is. This can increase the perceived size of the expression.
    """
    PARENT_CHILD = 1
    DAG = 2
    GRAPH = 3


def view_gem_dag(expressions, filename='gem_dag', directory=None, graph_type=GraphType.GRAPH, format='png'):
    """Create visual representation of the gem dag given by 'expressions'
    
    :arg expressions: a list of expressions
    :arg filename: filename to save the graph under
    :arg directory: directory to save the graph under
    :arg graph_type: the GraphType to create
    :arg format: output format of the graph. See https://www.graphviz.org/doc/info/output.html for options
    """
    print('\n')
    print('=== In view_gem_dag ===')
    g = Digraph(filename=filename, directory=directory, format=format)
    stack = []

    if graph_type == GraphType.PARENT_CHILD:
        for expr_part in expressions:
            stack.append(expr_part)
            parent = 'root'
            while stack:
                expr = stack.pop()
                g.edge(str(parent), str(expr))
                stack.extend(expr.children)
                parent = expr

    elif graph_type == GraphType.DAG:
        for expr_part in expressions:
            stack.append(expr_part)
            parent_id = 'root' 
            while stack:
                expr = stack.pop()
                expr_id = pretty_print_type(expr)
                g.edge(parent_id, expr_id, label='parent of')
                for slot in expr.__slots__:
                    if slot != 'children':
                        slot_id = str(getattr(expr, slot))
                        g.edge(expr_id, slot_id, label=slot)
                stack.extend(expr.children)
                parent_id = expr_id

    elif graph_type == GraphType.GRAPH:
        # Counter to make each expression and it's information unique
        i = 0
        for expr_part in expressions:
            stack.append(expr_part)
            parent_id = 'root' 
            while stack:
                expr = stack.pop()
                expr_id = '[' + str(i) + '] ' + pretty_print_type(expr)
                i += 1
                g.edge(parent_id, expr_id, label='parent of')
                for slot in expr.__slots__:
                    if slot != 'children':
                        slot_id = '[' + str(i) + ']' + str(getattr(expr, slot))
                        g.edge(expr_id, slot_id, label=slot)
                        i += 1
                stack.extend(expr.children)
                parent_id = expr_id
    else:
        raise NotImplementedError('Graph type ', graph_type, ' not supported')
    
    print('=== End view_gem_dag ===')
    print('\n')
    g.view()

def pretty_print_type(expression):
    expr = str(type(expression))
    try:
        class_name = search("(\w+?)'>", expr).group(1)
    except AttributeError:
        class_name = expr
    return class_name


