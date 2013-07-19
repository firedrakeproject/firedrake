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

save_soln_code = """
void save_soln(double *q, double *qold){
  for (int n=0; n<4; n++) qold[n] = q[n];
}
"""

adt_calc_code = """
void adt_calc(double *x[2], double *q,double *adt){
  double dx,dy, ri,u,v,c;

  ri =  1.0f/q[0];
  u  =   ri*q[1];
  v  =   ri*q[2];
  c  = sqrt(gam*gm1*(ri*q[3]-0.5f*(u*u+v*v)));

  dx = x[1][0] - x[0][0];
  dy = x[1][1] - x[0][1];
  *adt  = fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

  dx = x[2][0] - x[1][0];
  dy = x[2][1] - x[1][1];
  *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

  dx = x[3][0] - x[2][0];
  dy = x[3][1] - x[2][1];
  *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

  dx = x[0][0] - x[3][0];
  dy = x[0][1] - x[3][1];
  *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

  *adt = (*adt) / cfl;
}
"""

res_calc_code = """
void res_calc(double *x[2], double *q[4], double *adt[1], double *res[4]) {
  double dx,dy,mu, ri, p1,vol1, p2,vol2, f;

  dx = x[0][0] - x[1][0];
  dy = x[0][1] - x[1][1];

  ri   = 1.0f/q[0][0];
  p1   = gm1*(q[0][3]-0.5f*ri*(q[0][1]*q[0][1]+q[0][2]*q[0][2]));
  vol1 =  ri*(q[0][1]*dy - q[0][2]*dx);

  ri   = 1.0f/q[1][0];
  p2   = gm1*(q[1][3]-0.5f*ri*(q[1][1]*q[1][1]+q[1][2]*q[1][2]));
  vol2 =  ri*(q[1][1]*dy - q[1][2]*dx);

  mu = 0.5f*((adt[0][0])+(adt[1][0]))*eps;

  f = 0.5f*(vol1* q[0][0]         + vol2* q[1][0]        ) + mu*(q[0][0]-q[1][0]);
  res[0][0] += f;
  res[1][0] -= f;
  f = 0.5f*(vol1* q[0][1] + p1*dy + vol2* q[1][1] + p2*dy) + mu*(q[0][1]-q[1][1]);
  res[0][1] += f;
  res[1][1] -= f;
  f = 0.5f*(vol1* q[0][2] - p1*dx + vol2* q[1][2] - p2*dx) + mu*(q[0][2]-q[1][2]);
  res[0][2] += f;
  res[1][2] -= f;
  f = 0.5f*(vol1*(q[0][3]+p1)     + vol2*(q[1][3]+p2)    ) + mu*(q[0][3]-q[1][3]);
  res[0][3] += f;
  res[1][3] -= f;
}
"""

bres_calc_code = """
void bres_calc(double *x[2],  double *q1,
               double *adt1,double *res1,int *bound) {
  double dx,dy,mu, ri, p1,vol1, p2,vol2, f;

  dx = x[0][0] - x[1][0];
  dy = x[0][1] - x[1][1];

  ri = 1.0f/q1[0];
  p1 = gm1*(q1[3]-0.5f*ri*(q1[1]*q1[1]+q1[2]*q1[2]));

  if (*bound==1) {
    res1[1] += + p1*dy;
    res1[2] += - p1*dx;
  }
  else {
    vol1 =  ri*(q1[1]*dy - q1[2]*dx);

    ri   = 1.0f/qinf[0];
    p2   = gm1*(qinf[3]-0.5f*ri*(qinf[1]*qinf[1]+qinf[2]*qinf[2]));
    vol2 =  ri*(qinf[1]*dy - qinf[2]*dx);

    mu = (*adt1)*eps;

    f = 0.5f*(vol1* q1[0]         + vol2* qinf[0]        ) + mu*(q1[0]-qinf[0]);
    res1[0] += f;
    f = 0.5f*(vol1* q1[1] + p1*dy + vol2* qinf[1] + p2*dy) + mu*(q1[1]-qinf[1]);
    res1[1] += f;
    f = 0.5f*(vol1* q1[2] - p1*dx + vol2* qinf[2] - p2*dx) + mu*(q1[2]-qinf[2]);
    res1[2] += f;
    f = 0.5f*(vol1*(q1[3]+p1)     + vol2*(qinf[3]+p2)    ) + mu*(q1[3]-qinf[3]);
    res1[3] += f;
  }
}
"""

update_code = """
void update(double *qold, double *q, double *res, double *adt, double *rms){
  double del, adti;

  adti = 1.0f/(*adt);

  for (int n=0; n<4; n++) {
    del    = adti*res[n];
    q[n]   = qold[n] - del;
    res[n] = 0.0f;
    *rms  += del*del;
  }
}
"""

save_soln = Kernel(save_soln_code, "save_soln")
adt_calc = Kernel(adt_calc_code, "adt_calc")
res_calc = Kernel(res_calc_code, "res_calc")
bres_calc = Kernel(bres_calc_code, "bres_calc")
update = Kernel(update_code, "update")
