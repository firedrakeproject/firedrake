#ifndef FL_CFORTRAN_H
#define FL_CFORTRAN_H

#include "confdefs.h"

#define fl_fldopen_fc F77_FUNC_(fl_fldopen, FL_FLDOPEN)
#define fl_fldread_fc F77_FUNC_(fl_fldread, FL_FLDREAD)
#define fl_fldclose_fc F77_FUNC_(fl_fldclose, FL_FLDCLOSE)

#define mainfl_fc F77_FUNC(mainfl, MAINFL)

#define fprint_backtrace_fc F77_FUNC(fprint_backtrace, FPRINT_BACKTRACE)
#define fmain_fc F77_FUNC(fmain, FMAIN)
#define qg_strat_fc F77_FUNC(qg_strat, QG_STRAT)
#define get_elem_renumbering_fc F77_FUNC_(get_elem_renumbering, GET_ELEM_RENUMBERING)
#define addelm_fc F77_FUNC(addelm, ADDELM)
#define get_node_renumbering_fc F77_FUNC_(get_node_renumbering, GET_NODE_RENUMBERING)
#define addnode_fc F77_FUNC(addnode, ADDNODE)
#define set_dx_fc F77_FUNC_(set_dx, SET_DX)
#define resetmaps_fc F77_FUNC_(resetmaps, RESETMAPS)
#define drawqt_fc F77_FUNC(drawqt, DRAWQT)
#define flmkdir_fc F77_FUNC(flmkdir, FLMKDIR)
#define dprintf_fc F77_FUNC(dprintf, DPRINTF)
#define fldecomp_fc F77_FUNC(fldecomp, FLDECOMP)
#define gn_main_fc F77_FUNC_(gn_main, GN_MAIN)
#define set_global_debug_level_fc F77_FUNC_(set_global_debug_level, SET_GLOBAL_DEBUG_LEVEL)
#define set_pseudo2d_domain_fc F77_FUNC_(set_pseudo2d_domain, SET_PSEUDO2D_DOMAIN)
#define gallopede_main_fc F77_FUNC_(gallopede_main, GALLOPEDE_MAIN)
#define gn_operator_main_fc F77_FUNC_(gn_operator_main, GN_OPERATOR_MAIN)
#define ilink2_fc F77_FUNC(ilink2_legacy, ILINK2_LEGACY)
#define project_to_continuous_fc F77_FUNC(project_to_continuous, PROJECT_TO_CONTINUOUS)
#define vtk_read_file_fc F77_FUNC(vtk_read_file, VTK_READ_FILE)
#define vtk_get_sizes_fc F77_FUNC(vtk_get_sizes, VTK_GET_SIZES)

#endif
