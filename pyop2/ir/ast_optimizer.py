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
        self.itspace = []
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
                    if opts[2] == "itspace":
                        # Found high-level optimisation
                        self.itspace.append((node, parent))
                        return
                    delim = opts[2].find('(')
                    opt_name = opts[2][:delim].replace(" ", "")
                    opt_par = opts[2][delim:].replace(" ", "")
                    if opt_name == "outerproduct":
                        # Found high-level optimisation
                        # Store outer product iteration variables and parent
                        self.out_prods[node] = (
                            [opt_par[1], opt_par[3]], parent)
                    else:
                        raise RuntimeError("Unrecognised opt %s - skipping it", opt_name)
                else:
                    raise RuntimeError("Unrecognised pragma found '%s'", node.pragma)

        def inspect(node, parent, fors, decls, symbols):
            if isinstance(node, Block):
                self.block = node
                for n in node.children:
                    inspect(n, node, fors, decls, symbols)
                return (fors, decls, symbols)
            elif isinstance(node, For):
                check_opts(node, parent)
                fors.append((node, parent))
                return inspect(node.children[0], node, fors, decls, symbols)
            elif isinstance(node, Par):
                return inspect(node.children[0], node, fors, decls, symbols)
            elif isinstance(node, Decl):
                decls[node.sym.symbol] = node
                return (fors, decls, symbols)
            elif isinstance(node, Symbol):
                symbols.add(node)
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

        return inspect(node, self.pre_header, [], {}, set())

    def extract_itspace(self):
        """Remove fully-parallel loop from the iteration space. These are
        the loops that were marked by the user/higher layer with a 'pragma
        pyop2 itspace'."""

        itspace_vrs = []
        for node, parent in reversed(self.itspace):
            parent.children.extend(node.children[0].children)
            parent.children.remove(node)
            itspace_vrs.append(node.it_var())

        any_in = lambda a, b: any(i in b for i in a)
        accessed_vrs = [s for s in self.sym if any_in(s.rank, itspace_vrs)]

        return (itspace_vrs, accessed_vrs)
