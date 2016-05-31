/** Jenkinsfile for Firedrake 
 *  
 *  Place this file in a Firedrake branch so Jenkins knows to build that branch. This
 *  also requires a Dockerfile to be present.
 * 
 *  Original author: Tim Greaves <tim.greaves@imperial.ac.uk>
**/


/** Generate a random hash to identify this build **/
import java.util.UUID;
String build_hash = UUID.randomUUID().toString();

stage "Building"
node('firedrake-build') {
  env.BUILD_ID = build_hash;
  /** Get the source as specified by Jenkins following the build initialisation; there
   *  may be a bug here where it gets the *latest* revision as opposed to the one which
   *  triggered the build (this from reading bug reports on the Jenkins forum)           **/
  checkout scm;
    /** Build the docker image, using the Dockerfile in the repo and using HUDSON_COOKIE
   *  as a consistent and unique identifier for this build                               **/
  def build_cmd = $/docker build --build-arg FIREDRAKE_INSTALL_FLAGS="--minimal-petsc --disable-ssh --package-branch firedrake master" -t tmbgreaves/firedrakebuilds:$$BUILD_ID ./$
  sh build_cmd
  sh 'docker push tmbgreaves/firedrakebuilds:$BUILD_ID';
}

stage "Testing"

parallel (
    "stream 1" : { 
                     node('firedrake-test') { 
                           env.BUILD_ID = build_hash;
                           sh '''
                           docker run --env PYOP2_BACKEND='sequential' -a stdout -t tmbgreaves/firedrakebuilds:$BUILD_ID bash -c 'pip install pytest-xdist && py.test -n auto --color=no -v firedrake/src/firedrake/tests/'
                           ''';
                       } 
                   },
    "stream 2" : { 
                     node('firedrake-test') { 
                           env.BUILD_ID = build_hash;
                           sh '''
                           docker run --env OMP_NUM_THREADS='2' --env PYOP2_BACKEND='openmp' -a stdout -t tmbgreaves/firedrakebuilds:$BUILD_ID bash -c 'pip install pytest-xdist && py.test -n auto --color=no -v firedrake/src/firedrake/tests/'
                           ''';
                       } 
                   }
          )
