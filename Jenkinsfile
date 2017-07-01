pipeline {
  agent {
    label 'linux'
  }
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
            sh '$HOME/.local/bin/virtualenv --relocatable firedrake'
          }
        }
      }
    }
    stage('Lint'){
      steps {
        dir('tmp') {
          timestamps {
            sh '''
. ./firedrake/bin/activate
pip install flake8
cd firedrake/src/firedrake
make lint
'''
          }
        }
      }
    }
    stage('Test'){
      steps {
        dir('tmp') {
          timestamps {
            sh '''
. ./firedrake/bin/activate
export PYOP2_CACHE_DIR=${VIRTUAL_ENV}/pyop2_cache
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=${VIRTUAL_ENV}/tsfc_cache
firedrake-clean
pip install pytest-cov pytest-xdist
pip list
cd firedrake/src/firedrake
py.test -n 4 --cov firedrake -v tests
'''
          }
        }
      }
    }
    stage('Test Adjoint'){
      steps {
        dir('tmp') {
          timestamps {
            sh '''
. ./firedrake/bin/activate
export PYOP2_CACHE_DIR=${VIRTUAL_ENV}/pyop2_cache
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=${VIRTUAL_ENV}/tsfc_cache
cd firedrake/src/dolfin-adjoint; py.test -n 4 -v tests_firedrake
'''
          }
        }
      }
    }
    stage('Codecov'){
      steps {
        dir('tmp') {
          timestamps {
            sh '''
. ./firedrake/bin/activate
cd firedrake/src/firedrake
curl -s https://codecov.io/bash | bash
'''
          }
        }
      }
    }
  }
}

