pipeline {
  agent {
    label 'linux'
  }
  environment {
    PATH = "/usr/local/bin:/usr/bin:/bin"
    CC = "mpicc"
    FIREDRAKE_CI_TESTS = "1"
    PYTHONHASHSEED = "12453221"
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
            sh '../scripts/firedrake-install --disable-ssh --minimal-petsc ${SLEPC} --adjoint --slope --install thetis --install gusto ${PACKAGE_MANAGER} || (cat firedrake-install.log && /bin/false)'
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
python -m pip install flake8
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
python $(which firedrake-clean)
python -m pip install pytest-cov pytest-xdist
python -m pip list
cd firedrake/src/firedrake
python -m pytest -n 4 --cov firedrake -v tests
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
cd firedrake/src/dolfin-adjoint; python -m pytest -n 4 -v tests_firedrake
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

