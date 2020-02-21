from graphviz import Digraph

def view_gem_dag(expression):
    print('\n')
    print('=== In view_gem_dag ===')
    print(type(expression))
    print('expression: ', expression)
    g = Digraph('G', filename='gem_dag.gv')
    terms = []
    #stack = [expression]
    #while stack:
    #    expr = stack.pop()
    for i, expr in enumerate(expression):
        print('expr: ', expr)
        terms.append(expr)
        g.edge(str(expr), 'World')
        #stack.extend(expr.children)
    
    print('==terms: ', terms)
    print('=== End view_gem_dag ===')
    print('\n')
    g.view()