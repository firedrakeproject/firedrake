# Transform the kernel's ast depending on the backend we are executing over

from ir.ast_base import *


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

        for c in node.children:
            self._visit_ast(c, node, fors, decls)

        return (decls, fors)
