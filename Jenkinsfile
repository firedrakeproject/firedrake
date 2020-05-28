pipeline {
  agent {
    docker {
      image 'firedrakeproject/firedrake-env:latest'
      label 'firedrakeproject'
      args '-v /var/run/docker.sock:/var/run/docker.sock'
      alwaysPull true
    }
  }
  environment {
    FIREDRAKE_CI_TESTS = "1"
    DOCKER_CREDENTIALS = credentials('f52ccab9-5250-4b17-9fb6-c3f1ebdcc986')
    PETSC_CONFIGURE_OPTIONS = "--with-make-np=11"
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
            sh '../scripts/firedrake-install --disable-ssh --minimal-petsc --slepc --documentation-dependencies --install thetis --install gusto --install icepack --no-package-manager --package-branch tsfc reuse-create-domains --package-branch loopy allow-callablestables-gen-with-functionsnotinhistory --package-branch PyOP2 inverse-solve-callables|| (cat firedrake-install.log && /bin/false)'
          }
        }
      }
    }
    stage('Setup') {
      steps {
        dir('tmp') {
          timestamps {
            sh '''
. ./firedrake/bin/activate
python $(which firedrake-clean)
python -m pip install pytest-cov pytest-xdist
python -m pip list
'''
          }
        }
      }
    }
    stage('Test') {
      parallel {
        stage('Test Firedrake') {
          steps {
            dir('tmp') {
              timestamps {
                sh '''
. ./firedrake/bin/activate
cd firedrake/src/firedrake
python -m pytest --durations=200 -n 11 --cov firedrake -v tests
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
      }
    }
    stage('Post-test') {
      parallel {
        stage('Test build documentation') {
          steps {
            dir('tmp') {
              timestamps {
                sh '''
. ./firedrake/bin/activate
echo $PATH
echo $VIRTUAL_ENV
ls $VIRTUAL_ENV/bin
firedrake-preprocess-bibtex --validate firedrake/src/firedrake/docs/source/_static/bibliography.bib
firedrake-preprocess-bibtex --validate firedrake/src/firedrake/docs/source/_static/firedrake-apps.bib
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
        stage('Lint') {
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
      }
    }
    stage('Docker'){
      when {
        branch 'master'
      }
      steps {
        sh '''
sudo docker login -u $DOCKER_CREDENTIALS_USR -p $DOCKER_CREDENTIALS_PSW
sudo docker build -t firedrakeproject/firedrake-env:latest -f docker/Dockerfile.env .
sudo docker push firedrakeproject/firedrake-env:latest
sudo docker build --no-cache --build-arg PETSC_CONFIGURE_OPTIONS -t firedrakeproject/firedrake-vanilla:latest -f docker/Dockerfile.vanilla .
sudo docker push firedrakeproject/firedrake-vanilla:latest
sudo docker build --no-cache --build-arg PETSC_CONFIGURE_OPTIONS -t firedrakeproject/firedrake:latest -f docker/Dockerfile.firedrake .
sudo docker push firedrakeproject/firedrake:latest
sudo docker build --no-cache -t firedrakeproject/firedrake-notebooks:latest -f docker/Dockerfile.jupyter .
sudo docker push firedrakeproject/firedrake-notebooks:latest
'''
      }
    }
  }
}
