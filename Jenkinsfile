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
        sh 'git clean -fdx'
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
            sh '../scripts/firedrake-install --package-branch firedrake tsfc2loopy --package-branch tsfc tsfc2loopy --package-branch PyOP2 kaushik --disable-ssh --minimal-petsc ${SLEPC} --slope --documentation-dependencies --install thetis --install gusto --install icepack --install pyadjoint ${PACKAGE_MANAGER} || (cat firedrake-install.log && /bin/false)'
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
    stage('Test build documentation'){
      steps {
        dir('tmp') {
          timestamps {
            sh '''
. ./firedrake/bin/activate
echo $PATH
echo $VIRTUAL_ENV
ls $VIRTUAL_ENV/bin
cd firedrake/src/firedrake/docs; make html
'''
          }
        }
      }
    }
    stage('Zenodo API canary') {
      steps {
        timestamps {
          sh 'scripts/firedrake-install --test-doi-resolution || (cat firedrake-install.log && /bin/false)'
        }
      }
    }
  }
}

