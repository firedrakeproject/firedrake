import h5py
from math import sqrt
import numpy as np

from pyop2 import op2
from pyop2.configuration import configuration
from pyop2.fusion import loop_chain

from utils.benchmarking import parser, output_time

import slope_python


configuration['profiling'] = True


def main(args):

    num_unroll = args.num_unroll
    tile_size = args.tile_size
    part_mode = args.part_mode
    mesh_file = args.mesh_file

    save_soln, adt_calc, res_calc, bres_calc, update = kernels()

    try:
        with h5py.File(mesh_file, 'r') as f:

            # Declare sets, maps, datasets and global constants

            nodes = op2.Set.fromhdf5(f, "nodes")
            edges = op2.Set.fromhdf5(f, "edges")
            bedges = op2.Set.fromhdf5(f, "bedges")
            cells = op2.Set.fromhdf5(f, "cells")

            # The inspector needs to know this is actually a subset
            bedges_ss = op2.Subset(bedges, range(bedges.size))

            pedge = op2.Map.fromhdf5(edges, nodes, f, "pedge")
            pecell = op2.Map.fromhdf5(edges, cells, f, "pecell")
            pevcell = op2.Map.fromhdf5(edges, cells, f, "pecell")
            pbedge = op2.Map.fromhdf5(bedges, nodes, f, "pbedge")
            pbecell = op2.Map.fromhdf5(bedges, cells, f, "pbecell")
            pbevcell = op2.Map.fromhdf5(bedges, cells, f, "pbecell")
            pcell = op2.Map.fromhdf5(cells, nodes, f, "pcell")

            p_bound = op2.Dat.fromhdf5(bedges, f, "p_bound")
            p_x = op2.Dat.fromhdf5(nodes ** 2, f, "p_x")
            p_q = op2.Dat.fromhdf5(cells ** 4, f, "p_q")
            p_qold = op2.Dat.fromhdf5(cells ** 4, f, "p_qold")
            p_adt = op2.Dat.fromhdf5(cells, f, "p_adt")
            p_res = op2.Dat.fromhdf5(cells ** 4, f, "p_res")

            op2.Const.fromhdf5(f, "gam")
            op2.Const.fromhdf5(f, "gm1")
            op2.Const.fromhdf5(f, "cfl")
            op2.Const.fromhdf5(f, "eps")
            op2.Const.fromhdf5(f, "mach")
            op2.Const.fromhdf5(f, "alpha")
            op2.Const.fromhdf5(f, "qinf")

            # Tell SLOPE stuff about the mesh so that it can print it out
            slope_python.set_debug_mode('VERY_LOW',
                                        (p_x.dataset.set.name, p_x.data_ro, p_x.shape[1]))

    except IOError:
        import sys
        print "Failed reading mesh: Could not read from %s\n" % mesh_file
        sys.exit(1)

    # Main time-marching loop

    niter = 1000

    for i in range(1, niter + 1):

        with loop_chain("main", tile_size=tile_size, num_unroll=num_unroll, force_glb=True,
                        mode='only_tile', partitioning=part_mode):

            # Save old flow solution
            op2.par_loop(save_soln, cells,
                         p_q(op2.READ),
                         p_qold(op2.WRITE))

            # Predictor/corrector update loop
            for k in range(2):

                # Calculate area/timestep
                op2.par_loop(adt_calc, cells,
                             p_x(op2.READ, pcell[0]),
                             p_x(op2.READ, pcell[1]),
                             p_x(op2.READ, pcell[2]),
                             p_x(op2.READ, pcell[3]),
                             p_q(op2.READ),
                             p_adt(op2.WRITE))

                # Calculate flux residual
                op2.par_loop(res_calc, edges,
                             p_x(op2.READ, pedge[0]),
                             p_x(op2.READ, pedge[1]),
                             p_q(op2.READ, pevcell[0]),
                             p_q(op2.READ, pevcell[1]),
                             p_adt(op2.READ, pecell[0]),
                             p_adt(op2.READ, pecell[1]),
                             p_res(op2.INC, pevcell[0]),
                             p_res(op2.INC, pevcell[1]))

                op2.par_loop(bres_calc, bedges_ss,
                             p_x(op2.READ, pbedge[0]),
                             p_x(op2.READ, pbedge[1]),
                             p_q(op2.READ, pbevcell[0]),
                             p_adt(op2.READ, pbecell[0]),
                             p_res(op2.INC, pbevcell[0]),
                             p_bound(op2.READ))

                # Update flow field
                rms = op2.Global(1, 0.0, np.double, "rms")
                op2.par_loop(update, cells,
                             p_qold(op2.READ),
                             p_q(op2.WRITE),
                             p_res(op2.RW),
                             p_adt(op2.READ),
                             rms(op2.INC))

        # Print iteration history
        if i % 100 == 0:
            rms = sqrt(rms.data / cells.size)
            print " %d  %10.5e " % (i, rms)

    # Print runtime summary
    class FakeFunctionSpace():
        def __init__(self, dof_dset):
            self.dof_dset
    output_time(start, end,
                tofile=True,
                fs=FakeFunctionSpace(nodes),
                nloops=loop_chain_length * num_unroll,
                partitioning=part_mode,
                tile_size=tile_size)

def kernels():

    save_soln_code = """
    void save_soln(double *q, double *qold){
      for (int n=0; n<4; n++) qold[n] = q[n];
    }
    """

    adt_calc_code = """
    void adt_calc(double *x1,double *x2,double *x3,double *x4,double *q,double *adt){
      double dx,dy, ri,u,v,c;

      ri =  1.0f/q[0];
      u  =   ri*q[1];
      v  =   ri*q[2];
      c  = sqrt(gam*gm1*(ri*q[3]-0.5f*(u*u+v*v)));

      dx = x2[0] - x1[0];
      dy = x2[1] - x1[1];
      *adt  = fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

      dx = x3[0] - x2[0];
      dy = x3[1] - x2[1];
      *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

      dx = x4[0] - x3[0];
      dy = x4[1] - x3[1];
      *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

      dx = x1[0] - x4[0];
      dy = x1[1] - x4[1];
      *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

      *adt = (*adt) / cfl;
    }
    """

    res_calc_code = """
    void res_calc(double *x1,  double *x2,  double *q1,  double *q2,
                  double *adt1,double *adt2,double *res1,double *res2) {
      double dx,dy,mu, ri, p1,vol1, p2,vol2, f;

      dx = x1[0] - x2[0];
      dy = x1[1] - x2[1];

      ri   = 1.0f/q1[0];
      p1   = gm1*(q1[3]-0.5f*ri*(q1[1]*q1[1]+q1[2]*q1[2]));
      vol1 =  ri*(q1[1]*dy - q1[2]*dx);

      ri   = 1.0f/q2[0];
      p2   = gm1*(q2[3]-0.5f*ri*(q2[1]*q2[1]+q2[2]*q2[2]));
      vol2 =  ri*(q2[1]*dy - q2[2]*dx);

      mu = 0.5f*((*adt1)+(*adt2))*eps;

      f = 0.5f*(vol1* q1[0]         + vol2* q2[0]        ) + mu*(q1[0]-q2[0]);
      res1[0] += f;
      res2[0] -= f;
      f = 0.5f*(vol1* q1[1] + p1*dy + vol2* q2[1] + p2*dy) + mu*(q1[1]-q2[1]);
      res1[1] += f;
      res2[1] -= f;
      f = 0.5f*(vol1* q1[2] - p1*dx + vol2* q2[2] - p2*dx) + mu*(q1[2]-q2[2]);
      res1[2] += f;
      res2[2] -= f;
      f = 0.5f*(vol1*(q1[3]+p1)     + vol2*(q2[3]+p2)    ) + mu*(q1[3]-q2[3]);
      res1[3] += f;
      res2[3] -= f;
    }
    """

    bres_calc_code = """
    void bres_calc(double *x1,  double *x2,  double *q1,
                   double *adt1,double *res1,int *bound) {
      double dx,dy,mu, ri, p1,vol1, p2,vol2, f;

      dx = x1[0] - x2[0];
      dy = x1[1] - x2[1];

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

    save_soln = op2.Kernel(save_soln_code, "save_soln")
    adt_calc = op2.Kernel(adt_calc_code, "adt_calc")
    res_calc = op2.Kernel(res_calc_code, "res_calc")
    bres_calc = op2.Kernel(bres_calc_code, "bres_calc")
    update = op2.Kernel(update_code, "update")

    return save_soln, adt_calc, res_calc, bres_calc, update

if __name__ == '__main__':
    op2.init()
    main(parser())
