export TEST_FILES="tests/extrusion/test_facet_integrals_2D.py tests/extrusion/test_mixed_bcs.py tests/extrusion/test_steady_advection_2D_extr.py tests/multigrid/test_two_poisson_gmg.py tests/output tests/regression/test_facet_orientation.py tests/regression/test_matrix_free.py tests/regression/test_nested_fieldsplit_solves.py tests/regression/test_nullspace.py tests/regression/test_point_eval_api.py tests/regression/test_point_eval_cells.py tests/regression/test_point_eval_fs.py tests/regression/test_solving_interface.py tests/regression/test_steady_advection_2D.py"
export PATH=/usr/local/bin:$PATH
export CC=mpicc
export
#mkdir tmp
cd tmp
#../scripts/firedrake-install --disable-ssh --minimal-petsc ${SLEPC} --adjoint --slope --install thetis --install gusto ${PACKAGE_MANAGER}
. ./firedrake/bin/activate
pip install pytest-cov pytest-xdist
cd firedrake; py.test --cov firedrake --short -v ${TEST_FILES}
