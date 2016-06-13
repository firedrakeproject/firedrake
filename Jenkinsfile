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
  /** Build the docker image, using the Dockerfile in the repo, getting the SHA of the
   *  Firedrake commit to build from our local checkout, and using a generated random hash
   *  as a consistent and unique identifier for this build                               **/
  sh 'git rev-parse HEAD > local_commit_sha';
  env.LOCAL_COMMIT_SHA = readFile('local_commit_sha').trim();
  def build_cmd = $/docker build --build-arg FIREDRAKE_BUILD_SHA=$$LOCAL_COMMIT_SHA --build-arg FIREDRAKE_INSTALL_FLAGS="--minimal-petsc --disable-ssh --package-branch firedrake master" -t tmbgreaves/firedrakebuilds:$$BUILD_ID ./$;
  /**sh build_cmd;
  sh 'docker push tmbgreaves/firedrakebuilds:$BUILD_ID';**/
  /** See if the SHA for this build is also the SHA for a pull request; if it is, repeat
   *  the build for the merged pull request  **/
  def check_pr = $/git ls-remote origin | grep ^$$LOCAL_COMMIT_SHA | grep refs/pull | awk -F/ '{print $$3}' > local_pr_num/$;
  sh check_pr;
  env.LOCAL_PR = readFile('local_pr_num').trim();
  def get_merge_sha = $/git ls-remote origin | grep $$LOCAL_PR/merge$$ | awk -F/ '{if ($$3 ~ /^[0-9]+$$/) {print $$3;} else {print "null";} }' > local_merge_sha/$;
  sh get_merge_sha;
  def local_merge_sha = readFile('local_merge_sha').trim();
  env.LOCAL_MERGE_SHA = local_merge_sha;
  if ( local_merge_sha != 'null' ) {
    String build_merge_hash = UUID.randomUUID().toString();
    env.BUILD_MERGE_ID = build_merge_hash;
    def build_merge_cmd = $/docker build --build-arg FIREDRAKE_BUILD_SHA=$$LOCAL_MERGE_SHA --build-arg FIREDRAKE_INSTALL_FLAGS="--minimal-petsc --disable-ssh --package-branch firedrake master" -t tmbgreaves/firedrakebuilds:$$BUILD_MERGE_ID ./$;
    sh build_merge_cmd;
    sh 'docker push tmbgreaves/firedrakebuilds:$BUILD__MERGE_ID';
  }
}
/**
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
**/
