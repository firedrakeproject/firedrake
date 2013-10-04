#include "fdebug.h"

module linked_lists
  ! A module to provide linked lists and operations on them.
  use fldebug
  implicit none

  ! Define a linked list for integers
  TYPE inode
     INTEGER :: value
     TYPE (inode), POINTER :: next=>null() ! next node
  END TYPE inode

  TYPE ilist
     integer :: length=0
     TYPE (inode), POINTER :: firstnode=>null()
     type(inode), pointer ::  lastnode => null()
  END TYPE ilist

  ! <pef> I need a linked list for edges in the mesh.
  ! I'm adding it here.

  TYPE edgenode
     INTEGER :: i, j
     TYPE(edgenode), POINTER :: next => null()
  END TYPE edgenode

  TYPE elist
     INTEGER :: length = 0
     TYPE(edgenode), POINTER :: firstnode => null()
     TYPE(edgenode), POINTER :: lastnode => null()
  END TYPE elist

  ! <skramer> I need a linked list for reals
  ! I'm adding it here - sigh, templates anyone?

  type rnode
     real :: value
     type (rnode), pointer :: next=>null() ! next node
  end type rnode

  type rlist
     integer :: length=0
     type (rnode), pointer :: firstnode=>null()
     type(rnode), pointer ::  lastnode => null()
  end type rlist

  interface insert_ascending
     module procedure iinsert_ascending
  end interface

  interface has_value
     module procedure ihas_value, ehas_value
  end interface

  interface deallocate
    module procedure flush_ilist, flush_elist, flush_ilist_v, flush_rlist, flush_rlist_v
  end interface

  interface insert
     module procedure einsert, iinsert, rinsert
  end interface

  interface flush_list
     module procedure flush_ilist, flush_elist, flush_rlist
  end interface

  interface flush_lists
    module procedure flush_ilist_array
  end interface flush_lists

  interface pop
     module procedure ipop, epop_fn, rpop
  end interface

  interface fetch
     module procedure ifetch
  end interface

  interface spop ! I need this to be a subroutine, not a function
     module procedure epop
  end interface

  interface list2vector
     module procedure ilist2vector, rlist2vector
  end interface

  interface pop_last
    module procedure ipop_last
  end interface

  interface size_intersection
    module procedure isize_intersection
  end interface

  interface has_value_sorted
    module procedure ihas_value_sorted
  end interface

  interface print_list
    module procedure iprint, eprint
  end interface

  interface intersect_ascending
    module procedure intersect_ascending_ilist
  end interface intersect_ascending

  interface copy
     module procedure copy_ilist, copy_ilist_array
  end interface

  interface maxval
    module procedure list_maxval
  end interface maxval

contains

  integer function list_maxval(list)
    type(ilist), intent(in) :: list
    type(inode), pointer :: node

    node => list%firstnode
    list_maxval = node%value
    do while (associated(node))
      list_maxval = max(list_maxval, node%value)
      node => node%next
    end do
  end function list_maxval

  logical function ihas_value(list, value)
    ! Check if the list contains the value.
    type(ilist), intent(in) :: list
    integer, intent(in) :: value

    type(inode), pointer :: node

    ihas_value = .false.

    node => list%firstnode
    do while (associated(node))
       if(value==node%value) then
          ihas_value = .true.
          return
       end if
       node => node%next
    end do
  end function ihas_value

  subroutine iinsert_ascending(list, value, discard)
    ! Insert value in list in such a position as to ensure that list remains
    ! in ascending order. This assumes that list is in ascending order.
    ! Duplicate values are discarded!
    type(ilist), intent(inout) :: list
    integer, intent(in) :: value

    logical, optional :: discard

    type(inode), pointer :: this_node, next_node
    integer :: pos

    ! Special case for zero length lists.
    if (list%length==0) then
       allocate(list%firstnode)

       list%firstnode%value=value
       ! The following should not be necessary
       list%firstnode%next=>null()

       list%length=1
       return
    end if

    this_node=>list%firstnode
    next_node=>list%firstnode%next

    ! Special case for a value smaller than the first value.
    if (value<list%firstnode%value) then
       allocate(list%firstnode)

       list%firstnode%next=>this_node

       list%firstnode%value=value

       list%length=list%length+1
       return
    end if

    ! initialise discard logical
    if (present(discard)) discard =.false.

    do pos=0,list%length
       if(this_node%value==value) then
          ! Discard duplicates.
          if (present(discard)) discard = .true.
          return
       end if

       if (.not.associated(next_node)) then
          ! We have hit then end of the chain.
          allocate(this_node%next)

          if (this_node%value<value) then
             this_node%next%value=value
          else
             this_node%next%value=this_node%value
             this_node%value=value
          end if

          ! The following should not be necessary
          this_node%next%next=>null()

          list%length=list%length+1
          return
       end if

       ! Mid-chain. At this point we know this_node%value<value
       if (next_node%value>value) then
          ! Need to insert the value here.
          allocate(this_node%next)

          this_node%next%next=>next_node

          this_node%next%value=value

          list%length=list%length+1
          return
       end if

       ! Move along the chain.
       next_node=>next_node%next
       this_node=>this_node%next

    end do

    FLAbort("Walked off the end of the list. This can't happen.")

  end subroutine iinsert_ascending

  subroutine iinsert(list, i)
    type(ilist), intent(inout) :: list
    integer, intent(in) :: i
    type(inode), pointer :: node

    ! Special case for zero length lists.
    if (list%length==0) then
       allocate(list%firstnode)

       list%firstnode%value=i
       ! The following should not be necessary
       list%firstnode%next=>null()

       list%length=1
       list%lastnode => list%firstnode
       return
    end if

    node => list%lastnode
    allocate(node%next)
    node%next%value = i

    ! The following should not be necessary
    node%next%next => null()

    list%length = list%length+1
    list%lastnode => node%next
    return
  end subroutine iinsert

  subroutine flush_ilist(list)
    ! Remove all entries from a list.
    type(ilist), intent(inout) ::list

    integer :: i, tmp

    do i=1,list%length
       tmp=pop(list)
    end do

  end subroutine flush_ilist

  subroutine flush_ilist_v(lists)
    type(ilist), intent(inout), dimension(:) :: lists
    integer :: i

    do i=1,size(lists)
      call flush_ilist(lists(i))
    end do
  end subroutine flush_ilist_v

  subroutine flush_rlist_v(lists)
    type(rlist), intent(inout), dimension(:) :: lists
    integer :: i

    do i=1,size(lists)
      call flush_rlist(lists(i))
    end do
  end subroutine flush_rlist_v

  subroutine flush_ilist_array(lists)
    ! Remove all entries from an array of lists

    type(ilist), dimension(:), intent(inout) :: lists

    integer :: i

    do i = 1, size(lists)
      call flush_list(lists(i))
    end do

  end subroutine flush_ilist_array

  function ipop(list)
    ! Pop the first value off list.
    integer :: ipop
    type(ilist), intent(inout) :: list

    type(inode), pointer :: firstnode

    ipop=list%firstnode%value

    firstnode=>list%firstnode

    list%firstnode=>firstnode%next

    deallocate(firstnode)

    list%length=list%length-1

  end function ipop

  function ipop_last(list)
    ! Pop the last value off list.
    integer :: ipop_last
    type(ilist), intent(inout) :: list

    type(inode), pointer :: prev_node => null(), node
    integer :: i

    node => list%firstnode
    do i=1,list%length-1
      prev_node => node
      node => node%next
    end do

    ipop_last = node%value
    deallocate(node)
    prev_node%next => null()
    list%length = list%length - 1
  end function ipop_last

  function ifetch(list, j)
    integer :: ifetch
    type(ilist), intent(inout) :: list
    integer, intent(in) :: j

    type(inode), pointer :: node
    integer :: i

    node => list%firstnode
    do i=1,j-1
      node => node%next
    end do

    ifetch = node%value
  end function ifetch

  function ilist2vector(list) result (vector)
    ! Return a vector containing the contents of ilist
    type(ilist), intent(in) :: list
    integer, dimension(list%length) :: vector

    type(inode), pointer :: this_node
    integer :: i

    this_node=>list%firstnode

    do i=1,list%length
       vector(i)=this_node%value

       this_node=>this_node%next
    end do

  end function ilist2vector

  subroutine einsert(list, i, j)
    type(elist), intent(inout) :: list
    integer, intent(in) :: i, j

    ! Special case for zero length lists.
    if (list%length==0) then
       allocate(list%firstnode)

       list%firstnode%i=i
       list%firstnode%j=j
       ! The following should not be necessary
       list%firstnode%next=>null()

       list%length=1
       list%lastnode=>list%firstnode
       return
    end if


    allocate(list%lastnode%next)
    list%lastnode%next%i = i
    list%lastnode%next%j = j

    ! The following should not be necessary
    list%lastnode%next%next => null()

    list%length = list%length+1
    list%lastnode => list%lastnode%next
    return

  end subroutine einsert

  logical function ehas_value(list, i, j)
    type(elist), intent(inout) :: list
    integer, intent(in) :: i, j
    type(edgenode), pointer :: node

    ehas_value = .false.

    node => list%firstnode
    do while(associated(node))
      if (node%i == i .and. node%j == j) then
        ehas_value = .true.
        return
      end if
      node => node%next
    end do
  end function ehas_value

  subroutine flush_elist(list)
    ! Remove all entries from a list.
    type(elist), intent(inout) ::list

    integer :: i, tmp1, tmp2

    do i=1,list%length
       call spop(list, tmp1, tmp2)
    end do

  end subroutine flush_elist

  subroutine epop(list, i, j)
    ! Pop the first value off list.
    integer, intent(out) :: i, j
    type(elist), intent(inout) :: list

    type(edgenode), pointer :: firstnode

    i=list%firstnode%i
    j=list%firstnode%j

    firstnode=>list%firstnode

    list%firstnode=>firstnode%next

    deallocate(firstnode)

    list%length=list%length-1
  end subroutine epop

  function isize_intersection(listA, listB) result(x)
    type(ilist), intent(in) :: listA, listB
    type(inode), pointer :: nodeA, nodeB
    integer :: x

    x = 0
    nodeA => listA%firstnode
    do while(associated(nodeA))
      nodeB => listB%firstnode
      do while(associated(nodeB))
        if (nodeA%value == nodeB%value) then
          x = x + 1
          exit
        else
          nodeB => nodeB%next
        end if
      end do
      nodeA => nodeA%next
    end do
  end function isize_intersection

  function ihas_value_sorted(list, i) result(isin)
  ! This function assumes list is sorted
  ! in ascending order
    type(ilist), intent(in) :: list
    integer, intent(in) :: i
    type(inode), pointer :: node
    logical :: isin

    node => list%firstnode
    isin = .false.

    do while(associated(node))
      if (node%value > i) then
        return
      else if (node%value == i) then
        isin = .true.
        return
      end if
      node => node%next
    end do
  end function ihas_value_sorted

  function epop_fn(list) result(x)
    type(elist), intent(inout) :: list
    integer, dimension(2) :: x
    type(edgenode), pointer :: firstnode

    x(1) = list%firstnode%i
    x(2) = list%firstnode%j

    firstnode => list%firstnode
    list%firstnode => firstnode%next
    deallocate(firstnode)
    list%length = list%length - 1
  end function epop_fn

  subroutine iprint(list, priority)
    type(ilist), intent(in) :: list
    integer, intent(in) :: priority
    type(inode), pointer :: node

    ewrite(priority, *) "length: ", list%length

    node => list%firstnode
    do while (associated(node))
      ewrite(priority, *) " -- ", node%value
      node => node%next
    end do
  end subroutine

  subroutine eprint(list, priority)
    type(elist), intent(in) :: list
    integer, intent(in) :: priority
    type(edgenode), pointer :: node

    ewrite(priority, *) "length: ", list%length

    node => list%firstnode
    do while (associated(node))
      ewrite(priority, *) " -- (", node%i, ", ", node%j, ")"
      node => node%next
    end do
  end subroutine

  function intersect_ascending_ilist(list1, list2) result(intersection)
    !!< Assumes that list1 and list2 are already sorted
    type(ilist), intent(in) :: list1
    type(ilist), intent(in) :: list2

    type(ilist) :: intersection

    type(inode), pointer :: node1 => null(), node2 => null()

    node1 => list1%firstnode
    node2 => list2%firstnode
    do while(associated(node1) .and. associated(node2))
      if(node1%value == node2%value) then
        call insert_ascending(intersection, node1%value)
        node1 => node1%next
        node2 => node2%next
      else
        if(node1%value < node2%value) then
          node1 => node1%next
        else
          node2 => node2%next
        end if
      end if
    end do

  end function intersect_ascending_ilist

  subroutine copy_ilist(copy_list, list)
    !!< Make a deep copy of list
    type(ilist), intent(out) :: copy_list
    type(ilist), intent(in) :: list

    type(inode), pointer :: node, copy_node

    if (list%length==0) return

    ! Special case the first entry
    node=>list%firstnode
    allocate(copy_list%firstnode)
    copy_list%firstnode%value=node%value
    copy_node=>copy_list%firstnode
    copy_list%length=1
    node=>node%next

    do while(associated(node))
       allocate(copy_node%next)
       copy_node=>copy_node%next
       copy_node%value=node%value

       copy_list%length=copy_list%length+1

       node=>node%next
    end do

  end subroutine copy_ilist

  subroutine copy_ilist_array(copy_lists, lists)
    !!< Make a deep copy of list
    type(ilist), dimension(:), intent(in) :: lists
    type(ilist), dimension(size(lists)), intent(out) :: copy_lists

    integer :: i

    do i=1,size(lists)
       call copy_ilist(copy_lists(i), lists(i))
    end do

  end subroutine copy_ilist_array

    subroutine rinsert(list, value)
    type(rlist), intent(inout) :: list
    real, intent(in) :: value
    type(rnode), pointer :: node

    ! Special case for zero length lists.
    if (list%length==0) then
       allocate(list%firstnode)

       list%firstnode%value=value
       ! The following should not be necessary
       list%firstnode%next=>null()

       list%length=1
       list%lastnode => list%firstnode
       return
    end if

    node => list%lastnode
    allocate(node%next)
    node%next%value = value

    ! The following should not be necessary
    node%next%next => null()

    list%length = list%length+1
    list%lastnode => node%next

  end subroutine rinsert

  subroutine flush_rlist(list)
    ! Remove all entries from a list.
    type(rlist), intent(inout) ::list

    integer :: i, tmp

    do i=1,list%length
       tmp=pop(list)
    end do

  end subroutine flush_rlist

  function rpop(list)
    ! Pop the first value off list.
    real :: rpop
    type(rlist), intent(inout) :: list

    type(rnode), pointer :: firstnode

    rpop=list%firstnode%value

    firstnode=>list%firstnode

    list%firstnode=>firstnode%next

    deallocate(firstnode)

    list%length=list%length-1

  end function rpop

  function rlist2vector(list) result (vector)
    ! Return a vector containing the contents of rlist
    type(rlist), intent(in) :: list
    real, dimension(list%length) :: vector

    type(rnode), pointer :: this_node
    integer :: i

    this_node=>list%firstnode

    do i=1, list%length
       vector(i)=this_node%value

       this_node=>this_node%next
    end do

  end function rlist2vector

end module linked_lists
