subroutine test_submesh

  use elements
  use fields
  use fields_data_types
  use reference_counting
  use state_module
  use unittest_tools
  use fefields

  implicit none

  type(quadrature_type) :: quad
  type(element_type) :: baseshape, highshape
  type(mesh_type) :: basemesh, parmesh, highmesh, submesh1, submesh2
  type(vector_field) :: basex, parx, highx, subx1, subx2
  type(scalar_field) :: baselump, parlump, highlump, sublump1, sublump2

  integer :: i, j, dim, vertices

  do dim = 2,3
    write(0,*)
    write(0,*) 'dim = ', dim

    if(dim==3) then
      vertices = 4
    elseif(dim==2) then
      vertices = 3
    else
      write(0,*) "unsupported dimension count"
      return
    endif

    ! Make a P1 single triangle mesh
    quad = make_quadrature(vertices = vertices, dim  = dim, degree = 1)
    baseshape = make_element_shape(vertices = vertices, dim  = dim, degree = 1, quad = quad)
    call allocate(basemesh, nodes = vertices, elements = 1, shape = baseshape, name = "BaseMesh")
    call allocate(basex, mesh_dim(basemesh), basemesh, "BaseCoordinate")

    do i = 1, size(basemesh%ndglno)
      basemesh%ndglno(i) = i
    end do

    if(dim==3) then
      call set(basex, 1, (/0.0, 0.0, 0.0/))
      call set(basex, 2, (/1.0, 0.0, 0.0/))
      call set(basex, 3, (/0.0, sqrt(3.0)/2.0, 0.0/))
      call set(basex, 4, (/0.5, sqrt(3.0)/4.0, 0.75/))
    else
      call set(basex, 1, (/0.0, 0.0/))
      call set(basex, 2, (/1.0, 0.0/))
      call set(basex, 3, (/0.5, sqrt(3.0)/2.0/))
    end if

    call allocate(baselump, basemesh, "BaseLump")
    call compute_lumped_mass(basex, baselump)

    parmesh = make_mesh(basemesh, shape = baseshape, name="ParallelMesh")
    call allocate(parx, mesh_dim(parmesh), parmesh, "ParallelCoordinate")
    call remap_field(basex, parx)
    call allocate(parlump, parmesh, "ParallelLump")
    call compute_lumped_mass(parx, parlump)

    highshape = make_element_shape(vertices = vertices, dim  = dim, degree = 2, quad=quad)
    highmesh = make_mesh(basemesh, shape=highshape, name="HigherOrderMesh")
    call allocate(highx, mesh_dim(highmesh), highmesh, "HigherOrderCoordinate")
    call remap_field(basex, highx)
    call allocate(highlump, highmesh, "HigherOrderLump")
    call compute_lumped_mass(highx, highlump)

    submesh1 = make_submesh(parmesh, name="SubMesh1")
    call allocate(subx1, mesh_dim(submesh1), submesh1, name="SubCoordinate1")
    call set_to_submesh(parx, subx1)
    call allocate(sublump1, submesh1, "SubLump1")
    call compute_lumped_mass(subx1, sublump1)

    submesh2 = make_submesh(highmesh, name="SumMesh2")
    call allocate(subx2, mesh_dim(submesh2), submesh2, name="SubCoordinate2")
    call set_to_submesh(highx, subx2)
    call allocate(sublump2, submesh2, "SubLump1")
    call compute_lumped_mass(subx2, sublump2)

    write(0,*) "BaseMesh"
    do i = 1, basemesh%elements
      write(0,*) "Element ", i
      write(0,*) ele_nodes(basemesh, i)
      do j = 1, mesh_dim(basemesh)
        write(0,*) "dim ", j
        write(0,*) ele_val(basex, i, j)
      end do
    end do
    write(0,*) 'BaseLump'
    write(0,*) baselump%val
    write(0,*) 'sum = ', sum(baselump%val)
    write(0,*)

    write(0,*) "ParallelMesh"
    do i = 1, parmesh%elements
      write(0,*) "Element ", i
      write(0,*) ele_nodes(parmesh, i)
      do j = 1, mesh_dim(parmesh)
        write(0,*) "dim ", j
        write(0,*) ele_val(parx, i, j)
      end do
    end do
    write(0,*) 'ParLump'
    write(0,*) parlump%val
    write(0,*) 'sum = ', sum(parlump%val)
    write(0,*)

    write(0,*) "HigherOrderMesh"
    do i = 1, highmesh%elements
      write(0,*) "Element ", i
      write(0,*) ele_nodes(highmesh,i)
      do j = 1, mesh_dim(highmesh)
        write(0,*) "dim ", j
        write(0,*) ele_val(highx, i, j)
      end do
    end do
    write(0,*) 'HighLump'
    write(0,*) highlump%val
    write(0,*) 'sum = ', sum(highlump%val)
    write(0,*)

    write(0,*) "SubMesh1"
    do i = 1, submesh1%elements
      write(0,*) "Element ", i
      write(0,*) ele_nodes(submesh1,i)
      do j = 1, mesh_dim(submesh1)
        write(0,*) "dim ", j
        write(0,*) ele_val(subx1, i, j)
      end do
    end do
    write(0,*) 'SubLump1'
    write(0,*) sublump1%val
    write(0,*) 'sum = ', sum(sublump1%val)
    write(0,*)

    write(0,*) "SubMesh2"
    do i = 1, submesh2%elements
      write(0,*) "Element ", i
      write(0,*) ele_nodes(submesh2,i)
      do j = 1, mesh_dim(submesh2)
        write(0,*) "dim ", j
        write(0,*) ele_val(subx2, i, j)
      end do
    end do
    write(0,*) 'SubLump2'
    write(0,*) sublump2%val
    write(0,*) 'sum = ', sum(sublump2%val)
    write(0,*)

    call deallocate(quad)
    call deallocate(baseshape)
    call deallocate(highshape)
    call deallocate(basemesh)
    call deallocate(basex)
    call deallocate(parmesh)
    call deallocate(parx)
    call deallocate(highmesh)
    call deallocate(highx)
    call deallocate(submesh1)
    call deallocate(submesh2)
    call deallocate(subx1)
    call deallocate(subx2)
    call deallocate(sublump1)
    call deallocate(sublump2)
    call deallocate(parlump)
    call deallocate(highlump)
    call deallocate(baselump)
  end do

end subroutine test_submesh
