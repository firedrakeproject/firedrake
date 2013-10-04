module exodusii_f_interface

  use iso_c_binding
  implicit none

  private

  public :: f_read_ex_open, &
            f_ex_get_init, f_ex_get_coord, &
            f_ex_get_node_num_map, f_ex_get_elem_num_map, &
            f_ex_get_elem_order_map, f_ex_get_elem_blk_ids, &
            f_ex_get_elem_block, &
            f_ex_get_elem_block_parameters, &
            f_ex_get_elem_connectivity, &
            f_ex_get_node_set_param, f_ex_get_node_set_node_list, &
            f_ex_get_side_set_ids, f_ex_get_side_set_param, &
            f_ex_get_side_set, f_ex_get_side_set_node_list, &
            f_ex_close


  ! Open an ExodusII mesh file
  interface f_read_ex_open
    function c_read_ex_open(path, mode, comp_ws, io_ws, version) result(exoid) bind(c)
       use, intrinsic :: iso_c_binding
       implicit none
       character(kind=c_char, len=1):: path
       integer(kind=c_int) :: mode
       integer(kind=c_int) :: comp_ws
       integer(kind=c_int) :: io_ws
       real(kind=c_float) :: version
       integer(kind=c_int) :: exoid
     end function c_read_ex_open
  end interface

  ! Get database parameters from exodusII file
  interface f_ex_get_init
     function c_ex_get_init(exoid, title, num_dim, num_nodes, num_elem, &
                            num_elem_blk, num_node_sets, num_side_sets) &
                            result(error) bind(c)
       use, intrinsic :: iso_c_binding
       !implicit none
       integer(kind=c_int) :: exoid
       character(kind=c_char, len=1):: title
       integer(kind=c_int) :: num_dim
       integer(kind=c_int) :: num_nodes
       integer(kind=c_int) :: num_elem
       integer(kind=c_int) :: num_elem_blk
       integer(kind=c_int) :: num_node_sets
       integer(kind=c_int) :: num_side_sets
       integer(kind=c_int) :: error
     end function c_ex_get_init
  end interface

  ! Get coordinates of nodes:
  interface f_ex_get_coord
    function c_ex_get_coord(exoid, x, y, z) result(error) bind(c)
      use, intrinsic :: iso_c_binding
      integer(kind=c_int) :: exoid
      real(kind=c_float) :: x(*)
      real(kind=c_float) :: y(*)
      real(kind=c_float) :: z(*)
      integer(kind=c_int) :: error
    end function c_ex_get_coord
  end interface

  ! Get node number map
  interface f_ex_get_node_num_map
    function c_ex_get_node_num_map(exoid, node_map) result(error) bind(c)
      use, intrinsic :: iso_c_binding
      integer(kind=c_int) :: exoid
      integer(kind=c_int) :: node_map(*)
      integer(kind=c_int) :: error
    end function c_ex_get_node_num_map
  end interface

  ! Get element number map
  interface f_ex_get_elem_num_map
    function c_ex_get_elem_num_map(exoid, elem_num_map) result(error) bind(c)
      use, intrinsic :: iso_c_binding
      integer(kind=c_int) :: exoid
      integer(kind=c_int) :: elem_num_map(*)
      integer(kind=c_int) :: error
    end function c_ex_get_elem_num_map
  end interface

  ! Get element order map:
  interface f_ex_get_elem_order_map
    function c_ex_get_elem_order_map(exoid, elem_order_map) result(error) bind(c)
      use, intrinsic :: iso_c_binding
      integer(kind=c_int) :: exoid
      integer(kind=c_int) :: elem_order_map(*)
      integer(kind=c_int) :: error
    end function c_ex_get_elem_order_map
  end interface

  ! Get block ids:
  interface f_ex_get_elem_blk_ids
    function c_ex_get_elem_blk_ids(exoid, block_ids) result(error) bind(c)
      use, intrinsic :: iso_c_binding
      integer(kind=c_int) :: exoid
      integer(kind=c_int) :: block_ids(*)
      integer(kind=c_int) :: error
    end function c_ex_get_elem_blk_ids
  end interface

  interface f_ex_get_elem_block
    function c_ex_get_elem_block(exoid, block_id, elem_type, &
                                 num_elem_in_block, &
                                 num_nodes_per_elem, &
                                 num_attr) &
                                 result(error) bind(c)
      use, intrinsic :: iso_c_binding
      integer(kind=c_int) :: exoid
      integer(kind=c_int) :: block_id
      character(kind=c_char, len=1):: elem_type
      integer(kind=c_int) :: num_elem_in_block
      integer(kind=c_int) :: num_nodes_per_elem
      integer(kind=c_int) :: num_attr
      integer(kind=c_int) :: error
    end function c_ex_get_elem_block
  end interface

  ! Get block parameters
  interface f_ex_get_elem_block_parameters
    function c_ex_get_elem_block_parameters(exoid, num_elem_blk, &
                                            block_ids, num_elem_in_block, &
                                            num_nodes_per_elem) result(error) bind(c)
      use, intrinsic :: iso_c_binding
      integer(kind=c_int) :: exoid
      integer(kind=c_int) :: num_elem_blk
      integer(kind=c_int) :: block_ids(*)
      integer(kind=c_int) :: num_elem_in_block(*)
      integer(kind=c_int) :: num_nodes_per_elem(*)
      integer(kind=c_int) :: error
    end function c_ex_get_elem_block_parameters
  end interface

  interface f_ex_get_elem_connectivity
    function c_ex_get_elem_connectivity(exoid, block_id, elem_connectivity) result(error) bind(c)
      use, intrinsic :: iso_c_binding
      integer(kind=c_int) :: exoid
      integer(kind=c_int) :: block_id
      integer(kind=c_int) :: elem_connectivity(*)
      integer(kind=c_int) :: error
    end function c_ex_get_elem_connectivity
  end interface

  interface f_ex_get_node_set_param
    function c_ex_get_node_set_param(exoid, num_node_sets, node_set_ids, num_nodes_in_set) result(error) bind(c)
      use, intrinsic :: iso_c_binding
      integer(kind=c_int) :: exoid
      integer(kind=c_int) :: num_node_sets
      integer(kind=c_int) :: node_set_ids(*)
      integer(kind=c_int) :: num_nodes_in_set(*)
      integer(kind=c_int) :: error
    end function c_ex_get_node_set_param
  end interface

  interface f_ex_get_node_set_node_list
    function c_ex_get_node_set_node_list(exoid, num_node_sets, node_set_id, node_set_node_list) result(error) bind(c)
      use, intrinsic :: iso_c_binding
      integer(kind=c_int) :: exoid
      integer(kind=c_int) :: num_node_sets
      integer(kind=c_int) :: node_set_id
      integer(kind=c_int) :: node_set_node_list(*)
      integer(kind=c_int) :: error
    end function c_ex_get_node_set_node_list
  end interface

  interface f_ex_get_side_set_ids
    function c_ex_get_side_set_ids(exoid, side_set_ids) result(error) bind(c)
      use, intrinsic :: iso_c_binding
      integer(kind=c_int) :: exoid
      integer(kind=c_int) :: side_set_ids(*)
      integer(kind=c_int) :: error
    end function c_ex_get_side_set_ids
  end interface

  interface f_ex_get_side_set_param
    function c_ex_get_side_set_param(exoid, side_set_id, num_sides_in_set, num_df_in_set) result(error) bind(c)
      use, intrinsic :: iso_c_binding
      integer(kind=c_int) :: exoid
      integer(kind=c_int) :: side_set_id
      integer(kind=c_int) :: num_sides_in_set
      integer(kind=c_int) :: num_df_in_set
      integer(kind=c_int) :: error
    end function c_ex_get_side_set_param
  end interface

  interface f_ex_get_side_set
    function c_ex_get_side_set(exoid, side_set_id, elem_list, side_list) result(error) bind(c)
      use, intrinsic :: iso_c_binding
      integer(kind=c_int) :: exoid
      integer(kind=c_int) :: side_set_id
      integer(kind=c_int) :: elem_list(*)
      integer(kind=c_int) :: side_list(*)
      integer(kind=c_int) :: error
    end function c_ex_get_side_set
  end interface

  interface f_ex_get_side_set_node_list
    function c_ex_get_side_set_node_list(exoid, side_set_id, side_set_node_cnt_list, side_set_node_list) result(error) bind(c)
      use, intrinsic :: iso_c_binding
      integer(kind=c_int) :: exoid
      integer(kind=c_int) :: side_set_id
      integer(kind=c_int) :: side_set_node_cnt_list(*)
      integer(kind=c_int) :: side_set_node_list(*)
      integer(kind=c_int) :: error
    end function c_ex_get_side_set_node_list
  end interface

  ! Closing exodusII file
  interface f_ex_close
     function c_ex_close(exoid) result(ierr) bind(c)
       use, intrinsic :: iso_c_binding
       !implicit none
       integer(kind=c_int) :: exoid
       integer(kind=c_int) :: ierr
     end function c_ex_close
  end interface


end module exodusii_f_interface
