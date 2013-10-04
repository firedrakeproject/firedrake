!    Copyright (C) 2006 Imperial College London and others.
!
!    Please see the AUTHORS file in the main source directory for a full list
!    of copyright holders.
!
!    Prof. C Pain
!    Applied Modelling and Computation Group
!    Department of Earth Science and Engineering
!    Imperial College London
!
!    amcgsoftware@imperial.ac.uk
!
!    This library is free software; you can redistribute it and/or
!    modify it under the terms of the GNU Lesser General Public
!    License as published by the Free Software Foundation,
!    version 2.1 of the License.
!
!    This library is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
!    Lesser General Public License for more details.
!
!    You should have received a copy of the GNU Lesser General Public
!    License along with this library; if not, write to the Free Software
!    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
!    USA

#include "fdebug.h"
module AuxilaryOptions
contains

  SUBROUTINE HAVE_FS_TIDAL_OPTIONS(GOT_TIDAL)
    use FLDebug
    IMPLICIT NONE
    LOGICAL  GOT_TIDAL

    CHARACTER*4096 data_file

    data_file = ' '
    data_file(1:19) = 'FSoptions.dat'

    INQUIRE(file=data_file,exist=GOT_TIDAL)

    IF(GOT_TIDAL) then
       ewrite(3,*) 'Found a TIDAL option file - MDP',GOT_TIDAL
    else
       ewrite(3,*) 'Did not find a TIDAL option file - MDP',GOT_TIDAL
    end if
    return
  end SUBROUTINE HAVE_FS_TIDAL_OPTIONS

  ! DECIDES WHETHER TO APPLY SURFACE HEAT FLUXES, AND READS IN
  ! SOME CONTROLLING PARAMETERS
  SUBROUTINE HAVE_FS_EQTIDAL_OPTIONS(GOT_EQUIL_TIDE,WHICH_TIDE,prime_meridian,YOUR_SCFACTH0)
    use FLDebug
    IMPLICIT NONE
    LOGICAL  GOT_EQUIL_TIDE
    INTEGER  WHICH_TIDE(12)
    REAL    prime_meridian, YOUR_SCFACTH0
    CHARACTER*4096 data_file

    data_file = ' '
    data_file(1:19) = 'EQTDop.dat'

    INQUIRE(file=data_file,exist=GOT_EQUIL_TIDE)
    WHICH_TIDE(1:12) = 0
    IF(GOT_EQUIL_TIDE) then
       ewrite(3,*) 'Found an equilibrium TIDAL option file - MDP',GOT_EQUIL_TIDE
998    FORMAT(I9)
999    FORMAT(1E15.7)
       data_file = ' '
       data_file(1:13) = 'EQTDop.dat'

       OPEN(556,status='unknown',file=data_file)
       READ(556,998) WHICH_TIDE(1)
       READ(556,998) WHICH_TIDE(2)
       READ(556,998) WHICH_TIDE(3)
       READ(556,998) WHICH_TIDE(4)
       READ(556,998) WHICH_TIDE(5)
       READ(556,998) WHICH_TIDE(6)
       READ(556,998) WHICH_TIDE(7)
       READ(556,998) WHICH_TIDE(8)
       READ(556,998) WHICH_TIDE(9)
       READ(556,998) WHICH_TIDE(10)
       READ(556,998) WHICH_TIDE(11)
       READ(556,998) WHICH_TIDE(12)
       READ(556,999) prime_meridian
       READ(556,999) YOUR_SCFACTH0
       CLOSE(556)
       ewrite(3,*) 'WHICH_TIDE=',WHICH_TIDE
       ewrite(3,*) 'prime_meridian = ',prime_meridian
       ewrite(3,*) 'YOUR_SCFACTH0=',YOUR_SCFACTH0
       CLOSE(556)
    ELSE
       ewrite(3,*) 'Did not find an equilibrium TIDAL option file - MDP',GOT_EQUIL_TIDE
       prime_meridian = 0.0
       YOUR_SCFACTH0 = 1.0
    ENDIF
    RETURN
  END SUBROUTINE HAVE_FS_EQTIDAL_OPTIONS

end module AuxilaryOptions
