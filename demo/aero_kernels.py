# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

# This file contains code from the original OP2 distribution, in the code
# variables. The original copyright notice follows:

# Copyright (c) 2011, Mike Giles and others. Please see the AUTHORS file in
# the main source directory for a full list of copyright holders.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Mike Giles may not be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."

from pyop2.op2 import Kernel

dirichlet_code = """
void dirichlet(double *res){
    *res = 0.0;
}"""

dotPV_code = """
void dotPV(double *p, double*v, double *c) {
    *c += (*p)*(*v);
}"""

dotR_code = """
void dotR(double *r, double *c){
    *c += (*r)*(*r);
}"""

init_cg_code = """
void init_cg(double *r, double *c, double *u, double *v, double *p){
    *c += (*r)*(*r);
    *p = *r;
    *u = 0;
    *v = 0;
}"""

res_calc_code = """
void res_calc(double **x, double **phim, double *K, double **res) {
  for (int j = 0;j<4;j++) {
    for (int k = 0;k<4;k++) {
      OP2_STRIDE(K, j*4+k) = 0;
    }
  }
  for (int i = 0; i<4; i++) { //for each gauss point
  double det_x_xi = 0;
  double N_x[8];

  double a = 0;
  for (int m = 0; m<4; m++)
    det_x_xi += Ng2_xi[4*i+16+m]*x[m][1];
  for (int m = 0; m<4; m++)
    N_x[m] = det_x_xi * Ng2_xi[4*i+m];

  a = 0;
    for (int m = 0; m<4; m++)
    a += Ng2_xi[4*i+m]*x[m][0];
  for (int m = 0; m<4; m++)
    N_x[4+m] = a * Ng2_xi[4*i+16+m];

  det_x_xi *= a;

  a = 0;
  for (int m = 0; m<4; m++)
    a += Ng2_xi[4*i+m]*x[m][1];
  for (int m = 0; m<4; m++)
    N_x[m] -= a * Ng2_xi[4*i+16+m];

  double b = 0;
    for (int m = 0; m<4; m++)
    b += Ng2_xi[4*i+16+m]*x[m][0];
  for (int m = 0; m<4; m++)
    N_x[4+m] -= b * Ng2_xi[4*i+m];

  det_x_xi -= a*b;

    for (int j = 0;j<8;j++)
        N_x[j] /= det_x_xi;

    double wt1 = wtg2[i]*det_x_xi;
    //double wt2 = wtg2[i]*det_x_xi/r;

    double u[2] = {0.0, 0.0};
    for (int j = 0;j<4;j++) {
      u[0] += N_x[j]*phim[j][0];
      u[1] += N_x[4+j]*phim[j][0];
    }

    double Dk = 1.0 + 0.5*gm1*(m2-(u[0]*u[0]+u[1]*u[1]));
    double rho = pow(Dk,gm1i); //wow this might be problematic -> go to log?
    double rc2 = rho/Dk;

    for (int j = 0;j<4;j++) {
      res[j][0] += wt1*rho*(u[0]*N_x[j] + u[1]*N_x[4+j]);
    }
    for (int j = 0;j<4;j++) {
      for (int k = 0;k<4;k++) {
        OP2_STRIDE(K, j*4+k) += wt1*rho*(N_x[j]*N_x[k]+N_x[4+j]*N_x[4+k]) - wt1*rc2*(u[0]*N_x[j] + u[1]*N_x[4+j])*(u[0]*N_x[k] + u[1]*N_x[4+k]);
      }
    }
  }
}"""

spMV_code = """
void spMV(double **v, double *K, double **p){
  v[0][0] += OP2_STRIDE(K, 0) * p[0][0];
  v[0][0] += OP2_STRIDE(K, 1) * p[1][0];
  v[1][0] += OP2_STRIDE(K, 1) * p[0][0];
  v[0][0] += OP2_STRIDE(K, 2) * p[2][0];
  v[2][0] += OP2_STRIDE(K, 2) * p[0][0];
  v[0][0] += OP2_STRIDE(K, 3) * p[3][0];
  v[3][0] += OP2_STRIDE(K, 3) * p[0][0];
  v[1][0] += OP2_STRIDE(K, 4+1) * p[1][0];
  v[1][0] += OP2_STRIDE(K, 4+2) * p[2][0];
  v[2][0] += OP2_STRIDE(K, 4+2) * p[1][0];
  v[1][0] += OP2_STRIDE(K, 4+3) * p[3][0];
  v[3][0] += OP2_STRIDE(K, 4+3) * p[1][0];
  v[2][0] += OP2_STRIDE(K, 8+2) * p[2][0];
  v[2][0] += OP2_STRIDE(K, 8+3) * p[3][0];
  v[3][0] += OP2_STRIDE(K, 8+3) * p[2][0];
  v[3][0] += OP2_STRIDE(K, 15) * p[3][0];
}"""

update_code = """
void update(double *phim, double *res, double *u, double *rms){
    *phim -= *u;
    *res = 0.0;
    *rms += (*u)*(*u);
}"""

updateP_code = """
void updateP(double *r, double *p, const double *beta){
  *p = (*beta)*(*p)+(*r);
}"""

updateUR_code = """
void updateUR(double *u, double *r, double *p, double *v, const double *alpha){
  *u += (*alpha)*(*p);
  *r -= (*alpha)*(*v);
  *v = 0.0f;
}"""

dirichlet = Kernel(dirichlet_code, 'dirichlet')

dotPV = Kernel(dotPV_code, 'dotPV')

dotR = Kernel(dotR_code, 'dotR')

init_cg = Kernel(init_cg_code, 'init_cg')

res_calc = Kernel(res_calc_code, 'res_calc')

spMV = Kernel(spMV_code, 'spMV')

update = Kernel(update_code, 'update')

updateP = Kernel(updateP_code, 'updateP')

updateUR = Kernel(updateUR_code, 'updateUR')
