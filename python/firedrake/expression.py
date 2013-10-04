import ufl


class Expression(ufl.Coefficient):

    def __init__(self, code=None, element=None, cell=None, degree=None, **kwargs):

        self.code = code
        self.element = element
        self.cell = cell
        self.degree = degree
