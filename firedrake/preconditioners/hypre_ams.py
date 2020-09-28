from firedrake.petsc import PETSc
from firedrake import FunctionSpace, Constant, project, Interpolator, grad, TestFunction
from firedrake.assemble import allocate_matrix, create_assembly_callable

__all__ = ("HypreAMS")

class HypreAMS(firedrake.PCSNESBase):
    def initialize(self, obj):
        if isinstance(obj, PETSc.PC):
            A, P = obj.getOperators()
        elif isinstance(obj, PETSc.SNES):
            A, P = obj.ksp.pc.getOperators()
        else:
            raise ValueError("Not a PC or SNES?")

        prefix = obj.getOptionsPrefix()
        appctx = self.get_appctx(obj) 

        f = appctx['state']
        element = str(f.function_space().ufl_element().family()) #not sure this is right
        mesh = f.function_space().mesh()
        V = FunctionSpace(mesh, element, 1)

        print("element:", element)
        print("mesh:", mesh)
        print("f:", f)
        print("A:", A.getValuesCSR())
        print("P:", P.getValuesCSR())

        # build gradient matrix G
        P1 = FunctionSpace(mesh, "Lagrange", 1)
        Q = FunctionSpace(mesh, "N1curl", 1)
        G = Interpolator(grad(TestFunction(P1)), Q).callable().handle

        pc = PETSc.PC().create()
        pc.setOptionsPrefix(prefix + "HypreAMS_")
        pc.setOperators(A, P)

        pc.setType('hypre')
        pc.setHYPREType('ams')
        pc.setHYPREDiscreteGradient(G)
        zero_beta = PETSc.Options(prefix).getBool("pc_HypreAMS_zero_beta_poisson", default=False)
        if zero_beta == True:
            pc.setHYPRESetBetaPoissonMatrix(None)
#        pc.setCoordinates(np.reshape(
#            mesh._plex.getCoordinates().getArray(),
#            (-1,mesh.geometric_dimension())))

        # Build constants basis for the Nedelec space
        constants = []
        cvecs = []
        for i in range(3):
            direction = [1.0 if i == j else 0.0 for j in range(3)]
            c = project(Constant(direction), V)
            with c.vector().dat.vec_ro as cvec:
                cvecs.append(cvec)
        pc.setHYPRESetEdgeConstantVectors(cvecs[0], cvecs[1], cvecs[2])
        pc.setUp()

        self.pc = pc

    def apply(self, pc, x, y):
        self.pc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self.pc.applyTranspose(x, y)

    def view(self, pc, viewer=None):
        super(HypreAMS, self).view(pc, viewer)
        viewer.pushASCIITab()
        if hasattr(self, "pc"):
            viewer.printfASCII("PC to apply inverse\n")
            self.pc.view(viewer)
        viewer.popASCIITab()

    def update(self, pc):
        pass
