# Transform the kernel's ast depending on the backend we are executing over

from ast_base import *
from ast_optimizer import LoopOptimiser


class ASTKernel(object):

    """Transform a kernel. """

    def __init__(self, ast):
        self.ast = ast
        self.decl, self.fors = self._visit_ast(ast, fors=[], decls={})

    def _visit_ast(self, node, parent=None, fors=None, decls=None):
        """Return lists of:
            - declarations within the kernel
            - perfect loop nests
            - dense linear algebra blocks
        that will be exploited at plan creation time."""

        if isinstance(node, Decl):
            decls[node.sym.symbol] = node
            return (decls, fors)
        elif isinstance(node, For):
            fors.append((node, parent))
            return (decls, fors)
        elif isinstance(node, FunDecl):
            self.fundecl = node
        elif isinstance(node, (FlatBlock, PreprocessNode, Symbol)):
            return (decls, fors)

        for c in node.children:
            self._visit_ast(c, node, fors, decls)

        return (decls, fors)

    def plan_gpu(self):
        """Transform the kernel suitably for GPU execution. """

        lo = [LoopOptimiser(l, pre_l) for l, pre_l in self.fors]
        for nest in lo:
            itspace_vrs, accessed_vrs = nest.extract_itspace()

            for v in accessed_vrs:
                # Change declaration of non-constant iteration space-dependent
                # parameters by shrinking the size of the iteration space
                # dimension to 1
                decl = set(
                    [d for d in self.fundecl.args if d.sym.symbol == v.symbol])
                dsym = decl.pop().sym if len(decl) > 0 else None
                if dsym and dsym.rank:
                    dsym.rank = tuple([1 if i in itspace_vrs else j
                                       for i, j in zip(v.rank, dsym.rank)])

                # Remove indices of all iteration space-dependent and
                # kernel-dependent variables that are accessed in an itspace
                v.rank = tuple([0 if i in itspace_vrs and dsym else i
                                for i in v.rank])

            # Add iteration space arguments
            self.fundecl.args.extend([Decl("int", c_sym("%s" % i))
                                     for i in itspace_vrs])

        # Clean up the kernel removing variable qualifiers like 'static'
        for d in self.decl.values():
            d.qual = [q for q in d.qual if q not in ['static', 'const']]
