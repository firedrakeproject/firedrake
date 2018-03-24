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
            sh '../scripts/firedrake-install --disable-ssh --complex --minimal-petsc ${SLEPC} --adjoint --slope --install thetis --install gusto --install icepack --install pyadjoint ${PACKAGE_MANAGER} --package-branch ufl complex --package-branch tsfc complex --package-branch PyOP2 complex --package-branch COFFEE complex || (cat firedrake-install.log && /bin/false)'
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
cd firedrake/src/dolfin-adjoint; python -m pytest -n 4 -v tests_firedrake
'''
          }
        }
      }
    }
    stage('Test pyadjoint'){
      steps {
        dir('tmp') {
          timestamps {
            sh '''
. ./firedrake/bin/activate
cd firedrake/src/pyadjoint; python -m pytest -v tests/firedrake_adjoint
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

