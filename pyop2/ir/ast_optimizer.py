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
            - optimisations suggested by the higher layers via pragmas
            - ... """

        def check_opts(node, parent):
            """Check if node is associated some pragma. If that is the case,
            it saves this info so as to enable pyop2 optimising such node. """
            if node.pragma:
                opts = node.pragma.split(" ", 2)
                if len(opts) < 3:
                    return
                if opts[1] == "pyop2":
                    delim = opts[2].find('(')
                    opt_name = opts[2][:delim].replace(" ", "")
                    opt_par = opts[2][delim:].replace(" ", "")
                    # Found high-level optimisation
                    if opt_name == "outerproduct":
                        # Find outer product iteration variables and store the
                        # parent for future manipulation
                        self.out_prods[node] = ([opt_par[1], opt_par[3]], parent)
                    else:
                        # TODO: return a proper error
                        print "Unrecognised opt %s - skipping it", opt_name
                else:
                    # TODO: return a proper error
                    print "Unrecognised pragma - skipping it"

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
                check_opts(node, parent)
                inspect(node.children[0], node, fors, decls, symbols)
                inspect(node.children[1], node, fors, decls, symbols)
                return (fors, decls, symbols)
            else:
                return (fors, decls, symbols)

        return inspect(node, None, [], {}, [])
