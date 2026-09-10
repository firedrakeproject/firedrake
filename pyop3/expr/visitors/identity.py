import numbers

import pyop3.visitors


class ExpressionIdentityVisitor(pyop3.visitors.identity.IdentityVisitor):
    def __init__(self):
        super().__init__(
            shallow=True,
            allowed_types=pyop3.expr.Expression | numbers.Number,
        )
