import ufl
from pyadjoint.overloaded_type import create_overloaded_object, FloatingType
from pyadjoint.tape import annotate_tape, stop_annotating, get_working_tape, no_annotations
from firedrake.adjoint.blocks import FunctionAssignBlock, ProjectBlock, FunctionSplitBlock, FunctionMergeBlock
import firedrake


class FunctionMixin(FloatingType):

    @staticmethod
    def _ad_annotate_init(init):
        def wrapper(self, *args, **kwargs):
            FloatingType.__init__(self, *args,
                                  block_class=kwargs.pop("block_class", None),
                                  _ad_floating_active=kwargs.pop("_ad_floating_active", False),
                                  _ad_args=kwargs.pop("_ad_args", None),
                                  output_block_class=kwargs.pop("output_block_class", None),
                                  _ad_output_args=kwargs.pop("_ad_output_args", None),
                                  _ad_outputs=kwargs.pop("_ad_outputs", None), **kwargs)
            init(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def _ad_annotate_project(project):

        def wrapper(self, b, *args, **kwargs):

            annotate = annotate_tape(kwargs)

            with stop_annotating():
                output = project(self, b, *args, **kwargs)

            if annotate:
                bcs = kwargs.pop("bcs", [])
                block = ProjectBlock(b, self.function_space(), output, bcs)

                tape = get_working_tape()
                tape.add_block(block)

                block.add_output(output.create_block_variable())

            return output
        return wrapper

    @staticmethod
    def _ad_annotate_split(split):

        def wrapper(self, *args, **kwargs):
            annotate = annotate_tape(kwargs)
            with stop_annotating():
                output = split(self, *args, **kwargs)

            if annotate:
                output = tuple(firedrake.Function(output[i].function_space(),
                                                  output[i],
                                                  block_class=FunctionSplitBlock,
                                                  _ad_floating_active=True,
                                                  _ad_args=[self, i],
                                                  _ad_output_args=[i],
                                                  output_block_class=FunctionMergeBlock,
                                                  _ad_outputs=[self])
                               for i in range(len(output)))
            return output
        return wrapper

    @staticmethod
    def _ad_annotate_copy(copy):

        def wrapper(self, *args, **kwargs):
            annotate = annotate_tape(kwargs)
            func = copy(self, *args, **kwargs)

            if annotate:
                if kwargs.pop("deepcopy", False):
                    block = FunctionAssignBlock(func, self)
                    tape = get_working_tape()
                    tape.add_block(block)
                    block.add_output(func.create_block_variable())
                else:
                    # TODO: Implement. Here we would need to use floating types.
                    raise NotImplementedError("Currently kwargs['deepcopy'] must be set True")

            return func

        return wrapper

    @staticmethod
    def _ad_annotate_assign(assign):

        def wrapper(self, other, *args, **kwargs):
            """To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
            Firedrake assign call."""

            # do not annotate in case of self assignment
            annotate = annotate_tape(kwargs) and self != other

            if annotate:
                if not isinstance(other, ufl.core.operator.Operator):
                    other = create_overloaded_object(other)
                block = FunctionAssignBlock(self, other)
                tape = get_working_tape()
                tape.add_block(block)

            with stop_annotating():
                ret = assign(self, other, *args, **kwargs)

            if annotate:
                block.add_output(self.create_block_variable())

            return ret

        return wrapper

    def _ad_create_checkpoint(self):
        return self.copy(deepcopy=True)

    @no_annotations
    def _ad_convert_type(self, value, options=None):
        from firedrake import Function, TrialFunction, TestFunction, assemble

        options = {} if options is None else options
        riesz_representation = options.get("riesz_representation", "l2")

        if riesz_representation == "l2":
            return Function(self.function_space(), val=value)

        elif riesz_representation == "L2":
            ret = Function(self.function_space())
            u = TrialFunction(self.function_space())
            v = TestFunction(self.function_space())
            M = assemble(firedrake.inner(u, v)*firedrake.dx)
            firedrake.solve(M, ret, value)
            return ret

        elif riesz_representation == "H1":
            ret = Function(self.function_space())
            u = TrialFunction(self.function_space())
            v = TestFunction(self.function_space())
            M = assemble(firedrake.inner(u, v)*firedrake.dx
                         + firedrake.inner(firedrake.grad(u), firedrake.grad(v))*firedrake.dx)
            firedrake.solve(M, ret, value)
            return ret

        elif callable(riesz_representation):
            return riesz_representation(value)

        else:
            raise NotImplementedError(
                "Unknown Riesz representation %s" % riesz_representation)

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    @no_annotations
    def adj_update_value(self, value):
        self.original_block_variable.checkpoint = value._ad_create_checkpoint()

    @no_annotations
    def _ad_mul(self, other):
        from firedrake import Function

        r = Function(self.function_space())
        r.assign(self * other)
        return r

    @no_annotations
    def _ad_add(self, other):
        from firedrake import Function

        r = Function(self.function_space())
        Function.assign(r, self + other)
        return r

    def _ad_dot(self, other, options=None):
        from firedrake import assemble

        options = {} if options is None else options
        riesz_representation = options.get("riesz_representation", "l2")
        if riesz_representation == "l2":
            return self.vector().inner(other.vector())
        elif riesz_representation == "L2":
            return assemble(firedrake.inner(self, other)*firedrake.dx)
        elif riesz_representation == "H1":
            return assemble((firedrake.inner(self, other)
                            + firedrake.inner(firedrake.grad(self), other))*firedrake.dx)
        else:
            raise NotImplementedError(
                "Unknown Riesz representation %s" % riesz_representation)

    @staticmethod
    def _ad_assign_numpy(dst, src, offset):
        range_begin, range_end = dst.vector().local_range()
        m_a_local = src[offset + range_begin:offset + range_end]
        dst.vector().set_local(m_a_local)
        dst.vector().apply('insert')
        offset += dst.vector().size()
        return dst, offset

    @staticmethod
    def _ad_to_list(m):
        if not hasattr(m, "gather"):
            m_v = m.vector()
        else:
            m_v = m
        m_a = m_v.gather()

        return m_a.tolist()

    def _ad_copy(self):
        from firedrake import Function

        r = Function(self.function_space())
        r.assign(self)
        return r

    def _ad_dim(self):
        return self.function_space().dim()

    def _ad_imul(self, other):
        vec = self.vector()
        vec *= other

    def _ad_iadd(self, other):
        vec = self.vector()
        ovec = other.vector()
        if ovec.dat == vec.dat:
            vec *= 2
        else:
            vec += ovec

    def _reduce(self, r, r0):
        vec = self.vector().get_local()
        for i in range(len(vec)):
            r0 = r(vec[i], r0)
        return r0

    def _applyUnary(self, f):
        vec = self.vector()
        npdata = vec.get_local()
        for i in range(len(npdata)):
            npdata[i] = f(npdata[i])
        vec.set_local(npdata)

    def _applyBinary(self, f, y):
        vec = self.vector()
        npdata = vec.get_local()
        npdatay = y.vector().get_local()
        for i in range(len(npdata)):
            npdata[i] = f(npdata[i], npdatay[i])
        vec.set_local(npdata)

    def __deepcopy__(self, memodict={}):
        return self.copy(deepcopy=True)
