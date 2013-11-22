from pyop2.ir.ast_base import *


class LoopOptimiser(object):

    """ Loops optimiser:
        * LICM:
        * register tiling:
        * interchange: """

    def __init__(self, loop_nest, pre_header):
        self.loop_nest = loop_nest
        self.pre_header = pre_header
        self.out_prods = {}
        fors_loc, self.decls, self.sym = self._visit_nest(loop_nest)
        self.fors, self.for_parents = zip(*fors_loc)

    def _visit_nest(self, node):
        """Explore the loop nest and collect various info like:
            - which loops are in the nest
            - declarations
            - ...
        ."""

        def inspect(node, parent, fors, decls, symbols):
            if isinstance(node, Block):
                self.block = node
                for n in node.children:
                    inspect(n, node, fors, decls, symbols)
                return (fors, decls, symbols)
            elif isinstance(node, For):
                fors.append((node, parent))
                return inspect(node.children[0], node, fors, decls, symbols)
            elif isinstance(node, Par):
                return inspect(node.children[0], node, fors, decls, symbols)
            elif isinstance(node, Decl):
                decls[node.sym.symbol] = node
                return (fors, decls, symbols)
            elif isinstance(node, Symbol):
                if node.symbol not in symbols and node.rank:
                    symbols.append(node.symbol)
                return (fors, decls, symbols)
            elif isinstance(node, BinExpr):
                inspect(node.children[0], node, fors, decls, symbols)
                inspect(node.children[1], node, fors, decls, symbols)
                return (fors, decls, symbols)
            elif perf_stmt(node):
                inspect(node.children[0], node, fors, decls, symbols)
                inspect(node.children[1], node, fors, decls, symbols)
                return (fors, decls, symbols)
            else:
                return (fors, decls, symbols)

        return inspect(node, None, [], {}, [])
