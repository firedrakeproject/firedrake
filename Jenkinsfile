pipeline {
  agent any
  environment {
    TEST_FILES = "tests/extrusion/test_facet_integrals_2D.py tests/extrusion/test_mixed_bcs.py tests/extrusion/test_steady_advection_2D_extr.py tests/multigrid/test_two_poisson_gmg.py tests/output tests/regression/test_facet_orientation.py tests/regression/test_matrix_free.py tests/regression/test_nested_fieldsplit_solves.py tests/regression/test_nullspace.py tests/regression/test_point_eval_api.py tests/regression/test_point_eval_cells.py tests/regression/test_point_eval_fs.py tests/regression/test_solving_interface.py tests/regression/test_steady_advection_2D.py"
    PATH = "/usr/local/bin:/usr/bin:/bin"
    CC = "mpicc"
    FIREDRAKE_CI_TESTS = "1"
  }
  stages {
    stage('Clean') {
      steps {
        dir('tmp') {
          deleteDir()
        }
      }
    }
    stage('Build') {
      steps {
        sh 'mkdir tmp'
        dir('tmp') {
          timestamps {
            sh 'pip install virtualenv'
            sh '../scripts/firedrake-install --disable-ssh --minimal-petsc ${SLEPC} --adjoint --slope --install thetis --install gusto ${PACKAGE_MANAGER}'
          }
        }
      }
    }
    stage('Test'){
      steps {
        dir('tmp') {
          timestamps {
            sh """
. ./firedrake/bin/activate
firedrake-clean
pip install pytest-cov pytest-xdist
cd firedrake/src/firedrake; py.test --cov firedrake --short -v ${TEST_FILES}
"""
          }
        }
      }
    }
    stage('Test Adjoint'){
      steps {
        dir('tmp') {
          timestamps {
            sh """
. ./firedrake/bin/activate
cd firedrake/src/dolfin-adjoint; py.test -v tests_firedrake
"""
          }
        }
      }
    }
  }
}

