/* groovylint-disable NestedBlockDepth */
pipeline {
  agent none
  environment {
    FIREDRAKE_CI_TESTS = "1"
    DOCKER_CREDENTIALS = credentials('f52ccab9-5250-4b17-9fb6-c3f1ebdcc986')
    PETSC_CONFIGURE_OPTIONS = "--with-make-np=12 --download-mpich-device=ch3:sock"
  }
  stages {
    stage('BuildAndTest') {
      matrix {
        agent {
          docker {
            image 'firedrakeproject/firedrake-env:latest'
            label 'firedrakeproject'
            args '-v /var/run/docker.sock:/var/run/docker.sock'
            alwaysPull true
          }
        }
        axes {
          axis {
            name 'SCALAR_TYPE'
            values 'real', 'complex'
          }
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
            environment {
              COMPLEX = "${SCALAR_TYPE == "complex" ? "--complex" : ""}"
            }
            steps {
              sh 'mkdir tmp'
              dir('tmp') {
                timestamps {
                  sh '../scripts/firedrake-install $COMPLEX --tinyasm --disable-ssh --minimal-petsc --slepc --documentation-dependencies --install thetis --install gusto --install icepack --install irksome --no-package-manager || (cat firedrake-install.log && /bin/false)'
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
          stage('Test Firedrake') {
            steps {
              dir('tmp') {
                timestamps {
                  sh '''
      . ./firedrake/bin/activate
      cd firedrake/src/firedrake
      python -m pytest --durations=200 -n 12 --cov firedrake -v tests
      '''
                }
              }
            }
          }
          stage('Test pyadjoint'){
            when { environment name: 'SCALAR_TYPE', value: 'real' }
            steps {
              dir('tmp') {
                timestamps {
                  sh '''
      . ./firedrake/bin/activate
      cd firedrake/src/pyadjoint; python -m pytest --durations=200 -n 12 -v tests/firedrake_adjoint
      '''
                }
              }
            }
          }
          stage('Test build documentation') {
            when { environment name: 'SCALAR_TYPE', value: 'real' }
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
          stage('Docker'){
            when {
              allOf {
                branch 'master'
                environment name: 'SCALAR_TYPE', value: 'real'
              }
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
          stage('DockerComplex'){
            when {
              allOf {
                branch 'master'
                // Complex docker not working //
                expression { false }
                environment name: 'SCALAR_TYPE', value: 'complex'
              }
            }
            steps {
              sh '''
      sudo docker login -u $DOCKER_CREDENTIALS_USR -p $DOCKER_CREDENTIALS_PSW
      sudo docker build -t firedrakeproject/firedrake-env:latest -f docker/Dockerfile.env .
      sudo docker build --no-cache --build-arg PETSC_CONFIGURE_OPTIONS -t firedrakeproject/firedrake-complex:latest -f docker/Dockerfile.complex .
      sudo docker push firedrakeproject/firedrake-complex:latest
      '''
            }
          }
        }
      }
    }
  }
}
