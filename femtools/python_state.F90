!! Python states can be added to the dictionary via:
!!  python_add_state(State::S)
!! The last added state will be available as 'state', while all states added
!! are accessible via the 'states' dictionary.
!! Adding a state twice will result in overwriting the information. All states
!! are uniquely identified by their name attribute.

!! Files belonging to this module:
!! python_state.F90
!! python_statec.c
!! python_state_types.py

#include "fdebug.h"

module python_state
  use fldebug
  use quadrature
  use elements
  use fields
  use halos, only: halo_type
  use field_options, only: find_linear_parent_mesh
  use global_parameters, only:FIELD_NAME_LEN, current_debug_level, OPTION_PATH_LEN, PYTHON_FUNC_LEN
  use state_module
  use python_utils

  implicit none

  private

  public :: python_add_array, python_add_field
  public :: python_add_state, python_add_states, python_add_states_time
  public :: python_shell

  interface
    !! Add a state_type object into the Python interpreter
    subroutine python_add_statec(name,nlen)
      implicit none
      integer :: nlen
      character(len=nlen) :: name
    end subroutine python_add_statec
  end interface

  interface python_shell
     module procedure python_shell_state, python_shell_states
  end interface

  interface python_add_array
    subroutine python_add_array_double_1d(arr,sizex,name,name_len)
      implicit none
      integer :: name_len,sizex
      character(len=name_len) :: name
      real,dimension(sizex) :: arr
    end subroutine python_add_array_double_1d
    subroutine python_add_array_double_2d(arr,sizex,sizey,name,name_len)
      implicit none
      integer :: name_len,sizex,sizey
      character(len=name_len) :: name
      real,dimension(sizex,sizey) :: arr
    end subroutine python_add_array_double_2d
    subroutine python_add_array_double_3d(arr,sizex,sizey,sizez,name,name_len)
      implicit none
      integer :: name_len,sizex,sizey,sizez
      character(len=name_len) :: name
      real,dimension(sizex,sizey,sizez) :: arr
    end subroutine python_add_array_double_3d

    subroutine python_add_array_integer_1d(arr,sizex,name,name_len)
      implicit none
      integer :: name_len,sizex
      character(len=name_len) :: name
      integer,dimension(sizex) :: arr
    end subroutine python_add_array_integer_1d
    subroutine python_add_array_integer_2d(arr,sizex,sizey,name,name_len)
      implicit none
      integer :: name_len,sizex,sizey
      character(len=name_len) :: name
      integer,dimension(sizex,sizey) :: arr
    end subroutine python_add_array_integer_2d
    subroutine python_add_array_integer_3d(arr,sizex,sizey,sizez,name,name_len)
      implicit none
      integer :: name_len,sizex,sizey,sizez
      character(len=name_len) :: name
      integer,dimension(sizex,sizey,sizez) :: arr
    end subroutine python_add_array_integer_3d

    module procedure python_add_array_d_1d_directly
    module procedure python_add_array_d_2d_directly
    module procedure python_add_array_d_3d_directly

    module procedure python_add_array_i_1d_directly
    module procedure python_add_array_i_2d_directly
    module procedure python_add_array_i_3d_directly
  end interface python_add_array


  !! Add a field to a State (these are for the C-interface, python_add_field_directly() is what you want probably)
  interface python_add_field
    subroutine python_add_scalar(sx,x,name,nlen,field_type,option_path,oplen,state_name,snlen,&
      &mesh_name,mesh_name_len,uid)
      implicit none
      integer :: sx,nlen,field_type,oplen,snlen,mesh_name_len,uid
      real, dimension(sx) :: x
      character(len=nlen) :: name
      character(len=snlen) :: state_name
      character(len=oplen) :: option_path
      character(len=mesh_name_len) :: mesh_name
    end subroutine python_add_scalar

    subroutine python_add_csr_matrix_real(valuesSize, values, col_indSize, col_ind, row_ptrSize, &
      row_ptr, name, namelen, state_name,snlen, numCols)
      implicit none
      integer :: valuesSize,col_indSize,row_ptrSize,namelen,snlen,numCols
      real, dimension(valuesSize) :: values
      integer, dimension(col_indSize) :: col_ind
      integer, dimension(row_ptrSize) :: row_ptr
      character(len=namelen) :: name
      character(len=snlen) :: state_name
    end subroutine python_add_csr_matrix_real

    subroutine python_add_csr_matrix_integer(valuesSize, ivalues, col_indSize, col_ind, row_ptrSize, &
      row_ptr, name, namelen, state_name,snlen, numCols)
      implicit none
      integer :: valuesSize,col_indSize,row_ptrSize,namelen,snlen,numCols
      integer, dimension(valuesSize) :: ivalues
      integer, dimension(col_indSize) :: col_ind
      integer, dimension(row_ptrSize) :: row_ptr
      character(len=namelen) :: name
      character(len=snlen) :: state_name
    end subroutine python_add_csr_matrix_integer

    subroutine python_add_vector(numdim,sx,x,&
      &name,nlen,field_type,option_path,oplen,state_name,snlen,&
      &mesh_name,mesh_name_len,uid)
      implicit none
      integer :: sx,numdim,nlen,field_type,oplen,snlen,mesh_name_len,uid
      real, dimension(sx) :: x
      character(len=nlen) :: name
      character(len=snlen) :: state_name
      character(len=oplen) :: option_path
      character(len=mesh_name_len) :: mesh_name
    end subroutine python_add_vector
    subroutine python_add_tensor(sx,sy,sz,x,numdim,name,nlen,field_type,option_path,oplen,state_name,snlen,&
      &mesh_name,mesh_name_len,uid)
      implicit none
      integer :: sx,sy,sz,nlen,field_type,oplen,snlen,mesh_name_len,uid
      integer, dimension(2) :: numdim
      real, dimension(sx,sy,sz) :: x
      character(len=nlen) :: name
      character(len=snlen) :: state_name
      character(len=oplen) :: option_path
      character(len=mesh_name_len) :: mesh_name
    end subroutine python_add_tensor

    subroutine python_add_halo(name, name_len, nprocs, unn_offset, &
         state_name, state_name_len, comm, uid)
      implicit none
      integer :: name_len, nprocs, state_name_len, uid, comm, unn_offset
      character(len=name_len) :: name
      character(len=state_name_len) :: state_name
    end subroutine python_add_halo

    subroutine python_add_mesh(ndglno,sndglno,elements,element_classes,nodes,node_classes,name,nlen,parent_name,plen,&
      &option_path,oplen,continuity,region_ids,sregion_ids,state_name,state_name_len,halo_name,halo_name_len,element_halo_name,element_halo_name_len,uid)
      !! Add a mesh to the state called state_name
      implicit none
      integer, dimension(*) :: ndglno,region_ids    !! might cause a problem
      integer, dimension(4) :: element_classes, node_classes
      integer :: sndglno, elements, nodes, nlen, plen, oplen, continuity, sregion_ids, state_name_len, uid, halo_name_len, element_halo_name_len
      character(len=nlen) :: name
      character(len=plen) :: parent_name
      character(len=oplen) :: option_path
      character(len=state_name_len) :: state_name
      character(len=halo_name_len) :: halo_name
      character(len=element_halo_name_len) :: element_halo_name
    end subroutine python_add_mesh

    subroutine python_add_faces(state_name, state_name_len, mesh_name, mesh_name_len, surface_mesh_name,&
      &surface_mesh_name_len, surface_node_list, ssurface_node_list, face_element_list,&
      &sface_element_list, boundary_ids, sboundary_ids)
      !! Add a faces to the mesh called mesh_name in state state_name
      implicit none
      integer :: state_name_len, mesh_name_len, surface_mesh_name_len
      character(len=state_name_len) :: state_name
      character(len=mesh_name_len) :: mesh_name
      character(len=surface_mesh_name_len) :: surface_mesh_name
      integer, dimension(*) :: surface_node_list, face_element_list, boundary_ids
      integer :: ssurface_node_list, sface_element_list, sboundary_ids
    end subroutine python_add_faces

    subroutine python_add_element(dim,loc,ngi,degree,stname,slen,mname,mlen,n,nx,ny,dn,dnx,dny,dnz,&
      &size_spoly_x,size_spoly_y,size_dspoly_x,size_dspoly_y, family_name, family_name_len, &
      & type_name, type_name_len, &
      & coords, size_coords_x, size_coords_y)
      !! Add an element to the state with stname and mesh with mname
      implicit none
      integer :: dim,loc,ngi,degree,slen,mlen,nx,ny,dnx,dny,dnz, family_name_len, type_name_len
      integer :: size_spoly_x,size_spoly_y,size_dspoly_x,size_dspoly_y, size_coords_x, size_coords_y
      real,dimension(nx,ny) :: n
      real,dimension(dnx,dny,dnz) :: dn
      character(len=slen) :: stname
      character(len=mlen) :: mname
      character(len=family_name_len) :: family_name
      character(len=type_name_len) :: type_name
      real, dimension(size_coords_x,size_coords_y) :: coords
    end subroutine python_add_element

    subroutine python_add_quadrature(dim,loc,ngi,degree,weight,weight_size,locations,loc_size,surfacequad)
      !! Add a quadrature to the last added element
      implicit none
      integer :: weight_size, loc_size, dim,loc,ngi,degree
      integer :: surfacequad  !! Specifies whether this quadrature is the normal quadr. or surface_quadr.
      real, dimension(weight_size) :: weight
      real, dimension(loc_size) :: locations
    end subroutine python_add_quadrature

    subroutine python_add_polynomial(coefs, scoefs, degree, x,y, is_spoly)
      !! Add a polynomial to the last added element at position x,y
      !! is_spoly==1 <-> will be added to spoly, 0 to dspoly
      implicit none
      integer :: scoefs, degree, x,y,is_spoly
      real, dimension(scoefs) :: coefs
    end subroutine python_add_polynomial

    subroutine python_fetch_real_c(name, len, output)
      character(len=*), intent(in) :: name
      integer, intent(in) :: len
      real, intent(out) :: output
    end subroutine python_fetch_real_c

    module procedure python_add_scalar_directly
    module procedure python_add_vector_directly
    module procedure python_add_tensor_directly
    module procedure python_add_csr_matrix_directly
  end interface


  !! The function versions called in Fortran, mainly simplified arguments, then
  !! unwrapped and called to the interface to C
 contains

  subroutine python_add_scalar_directly(S,st)
    type(scalar_field) :: S
    type(state_type) :: st
    integer :: snlen,slen,oplen,mesh_name_len
    slen = len(trim(S%name))
    snlen = len(trim(st%name))
    oplen = len(trim(S%option_path))
    mesh_name_len = len(trim(S%mesh%name))
    call python_add_scalar(size(S%val,1),S%val,&
      trim(S%name),slen, S%field_type,S%option_path,oplen,trim(st%name),snlen,S%mesh%name,mesh_name_len,S%refcount%id)
  end subroutine python_add_scalar_directly

  subroutine python_add_csr_matrix_directly(csrMatrix,st)
    type(csr_matrix) :: csrMatrix
    type(state_type) :: st
    integer :: valSize, col_indSize, row_ptrSize, nameLen, statenameLen,numCols, valtype
    type(csr_sparsity) :: csrSparsity
    real, dimension(:), pointer :: values
    integer, dimension(:), pointer :: ivalues
    integer, dimension(:), pointer :: col_ind
    integer, dimension(:), pointer :: row_ptr

    csrSparsity = csrMatrix%sparsity
    values => csrMatrix%val
    ivalues => csrMatrix%ival

    if (associated(values)) then
      valtype = CSR_REAL
      valSize = size(csrMatrix%val,1)
    else
      valtype = CSR_INTEGER
      valSize = size(csrMatrix%ival,1)
    end if

    col_ind => csrSparsity%colm
    col_indSize = valSize
    row_ptr => csrSparsity%findrm
    row_ptrSize = size(csrSparsity%findrm,1)
    nameLen = len(trim(csrMatrix%name))
    statenameLen = len(trim(st%name))
    numCols = csrSparsity%columns
    if ( valtype == CSR_REAL ) then
       call python_add_csr_matrix_real(valSize, values, col_indSize, col_ind, row_ptrSize, row_ptr, &
            trim(csrMatrix%name), nameLen, trim(st%name),statenameLen,numCols)
    else
       call python_add_csr_matrix_integer(valSize, ivalues, col_indSize, col_ind, row_ptrSize, row_ptr, &
            trim(csrMatrix%name), nameLen, trim(st%name),statenameLen,numCols)
    end if
  end subroutine python_add_csr_matrix_directly

  subroutine python_add_vector_directly(V,st)
    type(vector_field) :: V
    type(state_type) :: st
    integer :: snlen,slen,oplen,mesh_name_len
    real, dimension(0), target :: zero

    slen = len(trim(V%name))
    snlen = len(trim(st%name))
    oplen = len(trim(V%option_path))
    mesh_name_len = len(trim(V%mesh%name))

    assert(v%dim==size(v%val,1))
    call python_add_vector(V%dim, size(V%val,2), V%val, &
      trim(V%name), slen, V%field_type, V%option_path, oplen,trim(st%name),snlen,V%mesh%name,mesh_name_len,V%refcount%id)

  end subroutine python_add_vector_directly

  subroutine python_add_tensor_directly(T,st)
    type(tensor_field) :: T
    type(state_type) :: st
    integer :: snlen,slen,oplen,mesh_name_len
    slen = len(trim(T%name))
    snlen = len(trim(st%name))
    oplen = len(trim(T%option_path))
    mesh_name_len = len(trim(T%mesh%name))
    call python_add_tensor(size(T%val,1),size(T%val,2),size(T%val,3),T%val, T%dim,&
      trim(T%name),slen, T%field_type,T%option_path,oplen,trim(st%name),snlen,T%mesh%name,mesh_name_len,T%refcount%id)
  end subroutine python_add_tensor_directly

  subroutine python_add_halo_directly(H, state)
    type(halo_type) :: H
    type(state_type) :: state
    character(len=100) :: tmp_name
    integer :: i

    do i = 1, H%nprocs
       write(tmp_name,'(a,i0)')'send', i
       call python_add_array(H%sends(i)%ptr, trim(tmp_name))
       write(tmp_name, '(a,i0)')'recv', i
       call python_add_array(H%receives(i)%ptr, trim(tmp_name))
    end do
    call python_add_array(H%receives_gnn_to_unn, 'receives_gnn2unn')
    call python_add_halo(trim(H%name), len(trim(H%name)), H%nprocs, &
         H%my_owned_nodes_unn_base, &
         trim(state%name), len(trim(state%name)), &
         H%communicator, H%refcount%id)
  end subroutine python_add_halo_directly

  subroutine python_add_mesh_directly(M,st)
    type(mesh_type) :: M
    type(mesh_type), pointer :: parent
    type(state_type) :: st
    integer :: snlen,slen,plen,oplen, stat
    integer, dimension(:), allocatable :: temp_region_ids
    integer :: halo_name_len, element_halo_name_len
    character(len=150) :: halo_name, element_halo_name
    call find_linear_parent_mesh(st, M, parent, stat)
    if ( stat /= 0 ) parent = M
    slen = len(trim(M%name))
    plen = len(trim(parent%name))
    snlen = len(trim(st%name))
    oplen = len(trim(M%option_path))

    if (associated(M%element_halos)) then
       call python_add_halo_directly(M%element_halos(halo_count(M)), st)
       element_halo_name_len = len(trim(M%element_halos(halo_count(M))%name))
       element_halo_name = trim(M%element_halos(halo_count(M))%name)
    else
       element_halo_name = ""
       element_halo_name_len = 0
    end if

    if (associated(M%halos)) then
       call python_add_halo_directly(M%halos(halo_count(M)), st)
       halo_name_len = len(trim(M%halos(halo_count(M))%name))
       halo_name = trim(M%halos(halo_count(M))%name)
    else
       halo_name = ""
       halo_name_len = 0
    end if

    if ( trim(parent%name) == trim(M%name) ) plen = 0
    if(associated(M%region_ids)) then
      call python_add_mesh(M%ndglno,size(M%ndglno,1),M%elements,&
        M%element_classes, M%nodes, M%node_classes,&
        trim(M%name),slen,trim(parent%name),plen,M%option_path,oplen,&
        M%continuity, M%region_ids, size(M%region_ids),&
        trim(st%name),snlen, &
        halo_name, halo_name_len,&
        element_halo_name, element_halo_name_len,&
        M%refcount%id)
    else
      allocate(temp_region_ids(0))
      call python_add_mesh(M%ndglno,size(M%ndglno,1),M%elements,&
        M%element_classes,M%nodes,M%node_classes,&
        trim(M%name),slen,trim(parent%name),plen,M%option_path,oplen,&
        M%continuity, temp_region_ids, size(temp_region_ids),&
        trim(st%name),snlen, &
        halo_name, halo_name_len,&
        element_halo_name, element_halo_name_len,&
        M%refcount%id)
      deallocate(temp_region_ids)
    end if

    if(associated(M%faces)) then
      call python_add_faces_directly(M%faces, M, st)
    end if

  end subroutine python_add_mesh_directly

  subroutine python_add_faces_directly(F,M,st)
    type(mesh_faces) :: F
    type(mesh_type) :: M
    type(state_type) :: st
    integer :: mlen, snlen, smlen

    mlen = len(trim(M%name))
    snlen = len(trim(st%name))
    smlen = len(trim(F%surface_mesh%name))

    call python_add_mesh_directly(F%surface_mesh, st)
    call python_add_element_directly(F%surface_mesh%shape,F%surface_mesh,st)

    call python_add_faces(trim(st%name), snlen, trim(M%name), mlen, &
                             & trim(F%surface_mesh%name), smlen, &
                             & F%surface_node_list, size(F%surface_node_list), &
                             & F%face_element_list, size(F%face_element_list), &
                             & F%boundary_ids, size(F%boundary_ids))

    call python_add_csr_matrix_directly(F%face_list, st)

    call python_run_string("states['" // trim(st%name) // "'].meshes['" // trim(M%name) // &
                         & "'].faces.face_list = states['"// trim(st%name) // &
                         & "'].csr_matrices['" // trim(M%faces%face_list%name) // &
                         & "']")

  end subroutine python_add_faces_directly

  subroutine python_add_element_directly(E,M,st)
    !! Add an element to the mesh M, by adding first the element and then its
    !! attributes one by one the element's
    !! 1) basic attributes
    !! 2) quadrature
    !! 3) spoly
    !! 4) dspoly
    type(element_type) :: E
    type(mesh_type) :: M
    type(state_type) :: st
    real, dimension(E%ndof, size(E%numbering%number2count, 1)) :: coords
    integer :: snlen,mlen
    integer :: i, j
    character(len=30) :: family_name, type_name
    integer :: l

    snlen = len(trim(st%name))
    mlen = len(trim(M%name))

    family_name = "unknown"
    if (cell_family(E) == FAMILY_SIMPLEX) then
      family_name = "simplex"
    else if (cell_family(E) == FAMILY_CUBE) then
      family_name = "cube"
    end if

    type_name = "unknown"
    if (E%type == ELEMENT_LAGRANGIAN) then
      type_name = "lagrangian"
   else if (E%type == ELEMENT_DISCONTINUOUS_LAGRANGIAN) then
      type_name = "discontinuous_lagrangian"
    else if (E%type == ELEMENT_BUBBLE) then
      type_name = "bubble"
    else if (E%type == ELEMENT_NONCONFORMING) then
      type_name = "nonconforming"
    end if

    do l=1,E%ndof
      coords(l,:) = local_coords(l, E)
    end do

    call python_add_element(E%dim, E%ndof, E%ngi, E%degree,&
      &trim(st%name),snlen,trim(M%name),mlen,&
      &E%n,size(E%n,1), size(E%n,2),E%dn, size(E%dn,1), size(E%dn,2), size(E%dn,3),&
      &size(E%spoly,1),size(E%spoly,2),size(E%dspoly,1),size(E%dspoly,2), family_name, len_trim(family_name), &
      &type_name, len_trim(type_name), &
      &coords, size(coords,1), size(coords,2))

    !! Add quadrature and surface_quadrature to this element
    call python_add_quadrature(E%quadrature%dim, E%quadrature%degree, E%quadrature%vertices,E%quadrature%ngi,&
      &E%quadrature%weight, size(E%quadrature%weight), &
      &E%quadrature%l, size(E%quadrature%l),0)
    if (associated(E%surface_quadrature)) then
      call python_add_quadrature(E%surface_quadrature%dim, E%surface_quadrature%degree, E%surface_quadrature%vertices,E%surface_quadrature%ngi,&
       &E%surface_quadrature%weight, size(E%surface_quadrature%weight), &
       &E%surface_quadrature%l, size(E%surface_quadrature%l),1)
    end if

    !! Since these are in an array, the polynomials must be added one by one, passing their indices
    if (associated(E%spoly)) then
      do i=1,size(E%spoly,1)
        do j=1,size(E%spoly,2)
          if(associated(E%spoly(i,j)%coefs)) then
            call python_add_polynomial(E%spoly(i,j)%coefs,size(E%spoly(i,j)%coefs),E%spoly(i,j)%degree,i,j,1)
          end if
        end do
      end do
    endif
    !! Do the same for dspoly
    if (associated(E%dspoly)) then
      do i=1,size(E%dspoly,1)
        do j=1,size(E%dspoly,2)
          if(associated(E%dspoly(i,j)%coefs)) then
            call python_add_polynomial(E%dspoly(i,j)%coefs,size(E%dspoly(i,j)%coefs),E%dspoly(i,j)%degree,i,j,0)
          end if
        end do
      end do
    endif
  end subroutine python_add_element_directly

  !! Insert a complete state into the python interpreter
  subroutine python_add_state(S)
    type(state_type) :: S
    integer :: i,nlen
    nlen = len(trim(S%name))
    call python_add_statec(trim(S%name),nlen)

    if ( associated(S%meshes) )  then
      do i=1,(size(S%meshes))
        call python_add_mesh_directly(S%meshes(i)%ptr,S)
        call python_add_element_directly(S%meshes(i)%ptr%shape,S%meshes(i)%ptr,S)
      end do
    end if
    if ( associated(S%scalar_fields) )  then
      do i=1,(size(S%scalar_fields))
        call python_add_field(S%scalar_fields(i)%ptr,S)
      end do
    end if
    if ( associated(S%vector_fields) )  then
      do i=1,(size(S%vector_fields))
        call python_add_field(S%vector_fields(i)%ptr,S)
      end do
    end if
    if ( associated(S%tensor_fields) )  then
      do i=1,(size(S%tensor_fields))
        call python_add_field(S%tensor_fields(i)%ptr,S)
      end do
    end if
    if ( associated(S%csr_matrices) )  then
      do i=1,(size(S%csr_matrices))
        call python_add_field(S%csr_matrices(i)%ptr,S)
      end do
    end if


  end subroutine python_add_state

  subroutine python_add_states(S)
    type(state_type), dimension(:) :: S
    integer :: i

    do i = 1, size(S)
       call python_add_state(S(i))
    end do

  end subroutine python_add_states

  subroutine python_add_states_time(S)
    type(state_type), dimension(:,:), intent(in), pointer :: S ! material_phases (1:n) x timesteps (p:q)
    integer :: min_timestep
    integer :: max_timestep
    integer :: i

    min_timestep = lbound(S, 2)
    max_timestep = ubound(S, 2)

    call python_run_string("megastates = [0] * " // int2str(max_timestep+1))
    do i=min_timestep,max_timestep
      call python_add_states(S(:, i))
      ! So right now, state = to the i'th state to be considered.
      ! Let's pack it into states[i-1]
      call python_run_string("megastates[" // int2str(i) // "] = states; states = {}")
    end do

    call python_run_string("states = megastates; del megastates; del state")

  end subroutine python_add_states_time

  subroutine python_shell_state(state)
    !!< Wrapper to allow python_shell to be called with a single state as
    !!< an argument.
    type(state_type), target, intent(inout) :: state

    type(state_type), dimension(1) :: states

    states(1)=state

    call python_shell_states(states)

  end subroutine python_shell_state

  subroutine python_shell_states(states)
    !!< Launch a python shell with access to the current state(s) provided. This is mostly
    !!< useful for debugging.

    type(state_type), dimension(:), target, intent(inout) :: states

#ifdef HAVE_NUMPY


    ! Clean up to make sure that nothing else interferes
    call python_reset()

    call python_add_states(states)

    call python_run_string("import fluidity_tools")

    call python_run_string("fluidity_tools.shell()()")

    ! Cleanup
    call python_reset()
#else
    FLExit("Python shell requires NumPy, which cannot be located.")
#endif

  end subroutine python_shell_states


  !! Wrapper procedures to add arrays to the Python interpreter

  subroutine python_add_array_d_1d_directly(arr,var_name)
    real,dimension(:) :: arr
    character(len=*) :: var_name
    integer :: name_len, sizex
    sizex = size(arr)
    name_len = len(var_name)
    call python_add_array_double_1d(arr,sizex,var_name,name_len)
  end subroutine python_add_array_d_1d_directly
  subroutine python_add_array_d_2d_directly(arr,var_name)
    real,dimension(:,:) :: arr
    character(len=*) :: var_name
    integer :: name_len, sizex, sizey
    sizex = size(arr,1)
    sizey = size(arr,2)
    name_len = len(var_name)
    call python_add_array_double_2d(arr,sizex,sizey,var_name,name_len)
  end subroutine python_add_array_d_2d_directly
  subroutine python_add_array_d_3d_directly(arr,var_name)
    real,dimension(:,:,:) :: arr
    character(len=*) :: var_name
    integer :: name_len, sizex, sizey, sizez
    sizex = size(arr,1)
    sizey = size(arr,2)
    sizez = size(arr,3)
    name_len = len(var_name)
    call python_add_array_double_3d(arr,sizex,sizey,sizez,var_name,name_len)
  end subroutine python_add_array_d_3d_directly

  subroutine python_add_array_i_1d_directly(arr,var_name)
    integer,dimension(:) :: arr
    character(len=*) :: var_name
    integer :: name_len, sizex
    sizex = size(arr)
    name_len = len(var_name)
    call python_add_array_integer_1d(arr,sizex,var_name,name_len)
  end subroutine python_add_array_i_1d_directly
  subroutine python_add_array_i_2d_directly(arr,var_name)
    integer,dimension(:,:) :: arr
    character(len=*) :: var_name
    integer :: name_len, sizex, sizey
    sizex = size(arr,1)
    sizey = size(arr,2)
    name_len = len(var_name)
    call python_add_array_integer_2d(arr,sizex,sizey,var_name,name_len)
  end subroutine python_add_array_i_2d_directly
  subroutine python_add_array_i_3d_directly(arr,var_name)
    integer,dimension(:,:,:) :: arr
    character(len=*) :: var_name
    integer :: name_len, sizex, sizey, sizez
    sizex = size(arr,1)
    sizey = size(arr,2)
    sizez = size(arr,3)
    name_len = len(var_name)
    call python_add_array_integer_3d(arr,sizex,sizey,sizez,var_name,name_len)
  end subroutine python_add_array_i_3d_directly

end module python_state
