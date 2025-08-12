Advanced tutorials
==================

These tutorials demonstrate some more advanced features of Firedrake's
PDE solving capabilities, such as block-preconditioning mixed finite
element systems.

.. toctree::
   :maxdepth: 1

   Printing in parallel.<demos/parprint.py>
   Benney-Luke nonlinear wave equation.<demos/benney_luke.py>
   Solving the one-layer Quasi-Geostrophic equations.<demos/qg_1layer_wave.py>
   Computing eigenmodes of the Quasi-Geostrophic equations using SLEPc.<demos/qgbasinmodes.py>
   A Quasi-Geostrophic wind driven gyre.<demos/qg_winddrivengyre.py>
   Preconditioning saddle-point systems, using the mixed Poisson problem as an example.<demos/saddle_point_systems.py>
   The Camassa-Holm equation, a nonlinear integrable PDE.<demos/camassaholm.py>
   The Monge-Ampère equation, a nonlinear PDE, demonstrating fieldsplit preconditioning.<demos/ma-demo.py>
   Preconditioning using geometric multigrid.<demos/geometric_multigrid.py>
   Linear mixed fluid-structure interaction system.<demos/linear_fluid_structure_interaction.py>
   Mass lumping for a high order spectral wave equation.<demos/higher_order_mass_lumping.py>
   Block preconditioning for the Stokes equation.<demos/stokes.py>
   A pressure-convection-diffusion preconditioner for the Navier-Stokes equations.</demos/navier_stokes.py>
   Rayleigh-Benard convection.<demos/rayleigh-benard.py>
   Netgen support.<demos/netgen_mesh.py>
   Full-waveform inversion: spatial and wave sources parallelism.<demos/full_waveform_inversion.py>
   Data assimilation: parallel-in-time solvers for weak constraint 4DVar.<demos/data_assimilation.py>
   1D Vlasov-Poisson equation using vertical independent function spaces.<demos/vp1d.py>
   Degree-independent multigrid convergence using patch relaxation.<demos/poisson_mg_patches.py>
   Monolithic multigrid with Vanka relaxation for Stokes.<demos/stokes_vanka_patches.py>
   Vertex/edge star multigrid relaxation for H(div).<demos/hdiv_riesz_star.py>
   Auxiliary space patch relaxation multigrid for H(curl).<demos/hcurl_riesz_star.py>
   Preconditioning using fast diagonalisation.<demos/fast_diagonalisation_poisson.py>
   Shape optimisation.<demos/shape_optimization.py>
   Steady Boussinesq problem with integral constraints.<demos/boussinesq.py>
   Steady multicomponent flow -- microfluidic mixing of hydrocarbons.<demos/multicomponent.py>
