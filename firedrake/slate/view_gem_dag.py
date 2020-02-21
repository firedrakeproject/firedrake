from graphviz import Digraph
from re import search

def view_gem_dag(expressions, filename=None, directory=None, condensed=False):
    """Create visual representation of the gem dag given by 'expressions'
    
    :arg expressions: a list of expressions
    :arg filename: filename to save the graph under
    :arg directory: directory to save the graph under
    :arg condensed: if true, graph is composed of parent-child relationships only and each node is labelled with all information about the node (its type and all its slots including 'children'). If false, the information about the node (i.e. its slots) are unique nodes in the graph.
    """
    print('\n')
    print('=== In view_gem_dag ===')
    if not filename:
        filename = 'gem_dag'
    format = 'png'
    g = Digraph(filename=filename, directory=directory)
    stack = []

    if condensed:
        for expr_part in expressions:
            stack.append(expr_part)
            parent = 'root'
            while stack:
                expr = stack.pop()
                g.edge(str(parent), str(expr))
                stack.extend(expr.children)
                parent = expr
    else:
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


