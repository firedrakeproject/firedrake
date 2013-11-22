# Transform the kernel's ast depending on the backend we are executing over

from ast_base import *
from ast_optimizer import LoopOptimiser


class ASTKernel(object):

    """Transform a kernel. """

    def __init__(self, ast):
        self.ast = ast
        self.decl, self.fors = self._visit_ast(ast)

    def _visit_ast(self, node, parent=None, fors=[], decls={}):
        """Return lists of:
            - declarations within the kernel
            - perfect loop nests
            - dense linear algebra blocks
        that will be exploited at plan creation time."""

        if isinstance(node, Decl):
            decls[node.sym.symbol] = node
            return (decls, fors)
        if isinstance(node, For):
            fors.append((node, parent))
            return (decls, fors)
        if isinstance(node, FunDecl):
            self.fundecl = node

        for c in node.children:
            self._visit_ast(c, node, fors, decls)

        return (decls, fors)

    def plan_gpu(self):
        """Transform the kernel suitably for GPU execution. """

        lo = [LoopOptimiser(l, pre_l) for l, pre_l in self.fors]
        for nest in lo:
            itspace_vars = nest.extract_itspace()
            self.fundecl.args.extend([Decl("int", c_sym("%s" % i)) for i in itspace_vars])

        # TODO: Need to change declaration of iteration space-dependent parameters
