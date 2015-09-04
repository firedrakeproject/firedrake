import numpy as np
import h5py
from math import sqrt
import os

from pyop2 import op2, utils


def main(opt):
    dirichlet, dotPV, dotR, init_cg, res_calc, spMV, update, updateP, updateUR = kernels()

    try:
        with h5py.File(opt['mesh'], 'r') as f:
            # sets
            nodes = op2.Set.fromhdf5(f, 'nodes')
            bnodes = op2.Set.fromhdf5(f, 'bedges')
            cells = op2.Set.fromhdf5(f, 'cells')

            # maps
            pbnodes = op2.Map.fromhdf5(bnodes, nodes, f, 'pbedge')
            pcell = op2.Map.fromhdf5(cells, nodes, f, 'pcell')
            pvcell = op2.Map.fromhdf5(cells, nodes, f, 'pcell')

            # dats
            p_xm = op2.Dat.fromhdf5(nodes ** 2, f, 'p_x')
            p_phim = op2.Dat.fromhdf5(nodes, f, 'p_phim')
            p_resm = op2.Dat.fromhdf5(nodes, f, 'p_resm')
            p_K = op2.Dat.fromhdf5(cells ** 16, f, 'p_K')
            p_V = op2.Dat.fromhdf5(nodes, f, 'p_V')
            p_P = op2.Dat.fromhdf5(nodes, f, 'p_P')
            p_U = op2.Dat.fromhdf5(nodes, f, 'p_U')
    except IOError:
        import sys
        print "Failed reading mesh: Could not read from %s\n" % opt['mesh']
        sys.exit(1)

    # Constants

    gam = 1.4
    gm1 = op2.Const(1, gam - 1.0, 'gm1', dtype=np.double)
    op2.Const(1, 1.0 / gm1.data, 'gm1i', dtype=np.double)
    op2.Const(2, [0.5, 0.5], 'wtg1', dtype=np.double)
    op2.Const(2, [0.211324865405187, 0.788675134594813], 'xi1',
              dtype=np.double)
    op2.Const(4, [0.788675134594813, 0.211324865405187,
                  0.211324865405187, 0.788675134594813],
              'Ng1', dtype=np.double)
    op2.Const(4, [-1, -1, 1, 1], 'Ng1_xi', dtype=np.double)
    op2.Const(4, [0.25] * 4, 'wtg2', dtype=np.double)
    op2.Const(16, [0.622008467928146, 0.166666666666667,
                   0.166666666666667, 0.044658198738520,
                   0.166666666666667, 0.622008467928146,
                   0.044658198738520, 0.166666666666667,
                   0.166666666666667, 0.044658198738520,
                   0.622008467928146, 0.166666666666667,
                   0.044658198738520, 0.166666666666667,
                   0.166666666666667, 0.622008467928146],
              'Ng2', dtype=np.double)
    op2.Const(32, [-0.788675134594813, 0.788675134594813,
                   -0.211324865405187, 0.211324865405187,
                   -0.788675134594813, 0.788675134594813,
                   -0.211324865405187, 0.211324865405187,
                   -0.211324865405187, 0.211324865405187,
                   -0.788675134594813, 0.788675134594813,
                   -0.211324865405187, 0.211324865405187,
                   -0.788675134594813, 0.788675134594813,
                   -0.788675134594813, -0.211324865405187,
                   0.788675134594813, 0.211324865405187,
                   -0.211324865405187, -0.788675134594813,
                   0.211324865405187, 0.788675134594813,
                   -0.788675134594813, -0.211324865405187,
                   0.788675134594813, 0.211324865405187,
                   -0.211324865405187, -0.788675134594813,
                   0.211324865405187, 0.788675134594813],
              'Ng2_xi', dtype=np.double)
    minf = op2.Const(1, 0.1, 'minf', dtype=np.double)
    op2.Const(1, minf.data ** 2, 'm2', dtype=np.double)
    op2.Const(1, 1, 'freq', dtype=np.double)
    op2.Const(1, 1, 'kappa', dtype=np.double)
    op2.Const(1, 0, 'nmode', dtype=np.double)
    op2.Const(1, 1.0, 'mfan', dtype=np.double)

    niter = 20

    for i in xrange(1, niter + 1):

        op2.par_loop(res_calc, cells,
                     p_xm(op2.READ, pvcell),
                     p_phim(op2.READ, pcell),
                     p_K(op2.WRITE),
                     p_resm(op2.INC, pcell))

        op2.par_loop(dirichlet, bnodes,
                     p_resm(op2.WRITE, pbnodes[0]))

        c1 = op2.Global(1, data=0.0, name='c1')
        c2 = op2.Global(1, data=0.0, name='c2')
        c3 = op2.Global(1, data=0.0, name='c3')
        # c1 = R' * R
        op2.par_loop(init_cg, nodes,
                     p_resm(op2.READ),
                     c1(op2.INC),
                     p_U(op2.WRITE),
                     p_V(op2.WRITE),
                     p_P(op2.WRITE))

        # Set stopping criteria
        res0 = sqrt(c1.data)
        res = res0
        res0 *= 0.1
        it = 0
        maxiter = 200

        while res > res0 and it < maxiter:

            # V = Stiffness * P
            op2.par_loop(spMV, cells,
                         p_V(op2.INC, pcell),
                         p_K(op2.READ),
                         p_P(op2.READ, pcell))

            op2.par_loop(dirichlet, bnodes,
                         p_V(op2.WRITE, pbnodes[0]))

            c2.data = 0.0

            # c2 = P' * V
            op2.par_loop(dotPV, nodes,
                         p_P(op2.READ),
                         p_V(op2.READ),
                         c2(op2.INC))

            alpha = op2.Global(1, data=c1.data / c2.data, name='alpha')

            # U = U + alpha * P
            # resm = resm - alpha * V
            op2.par_loop(updateUR, nodes,
                         p_U(op2.INC),
                         p_resm(op2.INC),
                         p_P(op2.READ),
                         p_V(op2.RW),
                         alpha(op2.READ))

            c3.data = 0.0
            # c3 = resm' * resm
            op2.par_loop(dotR, nodes,
                         p_resm(op2.READ),
                         c3(op2.INC))

            beta = op2.Global(1, data=c3.data / c1.data, name="beta")
            # P = beta * P + resm
            op2.par_loop(updateP, nodes,
                         p_resm(op2.READ),
                         p_P(op2.RW),
                         beta(op2.READ))

            c1.data = c3.data
            res = sqrt(c1.data)
            it += 1

        rms = op2.Global(1, data=0.0, name='rms')

        # phim = phim - Stiffness \ Load
        op2.par_loop(update, nodes,
                     p_phim(op2.RW),
                     p_resm(op2.WRITE),
                     p_U(op2.READ),
                     rms(op2.INC))

        print "rms = %10.5e iter: %d" % (sqrt(rms.data) / sqrt(nodes.size), it)


def kernels():

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

    dirichlet = op2.Kernel(dirichlet_code, 'dirichlet')
    dotPV = op2.Kernel(dotPV_code, 'dotPV')
    dotR = op2.Kernel(dotR_code, 'dotR')
    init_cg = op2.Kernel(init_cg_code, 'init_cg')
    res_calc = op2.Kernel(res_calc_code, 'res_calc')
    spMV = op2.Kernel(spMV_code, 'spMV')
    update = op2.Kernel(update_code, 'update')
    updateP = op2.Kernel(updateP_code, 'updateP')
    updateUR = op2.Kernel(updateUR_code, 'updateUR')

    return dirichlet, dotPV, dotR, init_cg, res_calc, spMV, update, updateP, updateUR

if __name__ == '__main__':
    parser = utils.parser(group=True, description=__doc__)
    parser.add_argument('-m', '--mesh', default='meshes/FE_grid.h5',
                        help='HDF5 mesh file to use (default: meshes/FE_grid.h5)')
    parser.add_argument('-p', '--profile', action='store_true',
                        help='Create a cProfile for the run')
    opt = vars(parser.parse_args())
    op2.init(**opt)

    if opt['profile']:
        import cProfile
        filename = 'aero.%s.cprofile' % os.path.split(opt['mesh'])[-1]
        cProfile.run('main(opt)', filename=filename)
    else:
        main(opt)
