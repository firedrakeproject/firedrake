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
module adjacency_lists
  ! **********************************************************************
  !! Module to construct mesh adjacency lists.
  use sparse_tools
  use FLDebug
  use fields_base
  use fields_data_types
  use element_numbering
  use futils
  use integer_set_module
  implicit none

  interface MakeLists
     module procedure MakeLists_Dynamic, MakeLists_Mesh
  end interface

contains


SUBROUTINE NODELE(NONODS,FINDELE,COLELE, &
     NCOLEL,MXNCOLEL, &
     TOTELE,NLOC,NDGLNO)

  Implicit None

  ! This sub calculates the node to element list FINDELE,COLELE
  INTEGER, Intent(In)::NONODS, MXNCOLEL
  INTEGER, Intent(Out)::NCOLEL,COLELE(MXNCOLEL),FINDELE(NONODS+1)
  INTEGER, intent(in)::TOTELE,NLOC,NDGLNO(TOTELE*NLOC)

  ! Local variables...
  INTEGER NOD,ELE,ILOC,COUNT, iNod
  Integer, Allocatable::NList(:), InList(:)

  Allocate(NList(Nonods))
  Allocate(InList(Nonods))

  DO NOD=1,NONODS
     NLIST(NOD)=0
     INLIST(NOD)=0
  END DO

  DO ELE=1,TOTELE
     DO ILOC=1,NLOC
        INOD=NDGLNO((ELE-1)*NLOC+ILOC)
        NLIST(INOD)=NLIST(INOD)+1   ! number of elements inod is part of
     END DO
  END DO

  COUNT=0
  DO NOD=1,NONODS
     FINDELE(NOD)=COUNT+1
     COUNT=COUNT+NLIST(NOD)
  END DO
  FINDELE(NONODS+1)=COUNT+1
  NCOLEL=COUNT

  DO ELE=1,TOTELE
     DO ILOC=1,NLOC
        INOD=NDGLNO((ELE-1)*NLOC+ILOC)
        INLIST(INOD)=INLIST(INOD)+1
        COLELE(FINDELE(INOD)-1+INLIST(INOD))=ELE
     END DO
  END DO

  Deallocate(NList, InList)

  RETURN
END SUBROUTINE NODELE









  ! **************************************************************************

  Subroutine MakeEEList_old(Nonods,Totele,NLoc,&
       D3, &
       ENList, &
       NEList, lgthNEList, NEListBasePtr,  &
       EEList)

    !! This subroutine returns ordered list of elements connected to 'ele'
    !! Zero entry indicates a free boundary, ie. on the surface

    Implicit None

    Integer, Intent(In)::Nonods, Totele, NLoc
    Logical, Intent(In)::D3
    Integer, Intent(In)::ENList(totele*nloc)
    Integer, intent(In)::lgthNEList, NEList(lgthNElist), NEListBasePtr(Nonods+1)

    Integer, Intent(Out)::EEList(Totele*NLoc)

    Integer, Allocatable::LocalElements(:), LocalNodes(:), n(:)
    ! Local...
    Integer Ele, ILoc, EEmark, ErrMark
    Integer ComEle, OppNode, i, p, OtherNode

    ewrite(3, *) "Inside Subroutine MakeEEList_old"

    Allocate(LocalElements(10000))

    Allocate(LocalNodes(NLoc))
    Allocate(n(NLoc-1))


    !Check NElist before we start
    do i=1,lgthNEList
       if(NEList(i).eq.0) then
          ewrite(-1,*) 'NEList zero:',i,NEList(i)
          FLAbort("Dieing")
       end if
    end do


    EEmark = 1

    Do ele=1,Totele

       do iLoc=1,NLoc
          LocalNodes(iLoc) = ENList((ele-1)*nloc+iLoc)
       end do

       Call LocalElementsNods(Nonods, Nloc, &
            NEList, NEListBasePtr, lgthNEList, &
            LocalNodes, &
            10000, LocalElements, p)

       ErrMark = 0

       if(D3) Then

          OppNode = LocalNodes(1)
          n(1) = LocalNodes(2)
          n(2) = LocalNodes(3)
          n(3) = LocalNodes(4)

          Call match_list(Totele, Nloc, &
               D3, &
               ENList, &
               OppNode, n, &
               LocalElements, p, &
               ComEle, OtherNode)

          EEList(EEmark) = ComEle
          EEmark = EEmark + 1

          if(ComEle.eq.0) then
             ErrMark = ErrMark+1
          end if

          OppNode = LocalNodes(2)
          n(1) = LocalNodes(3)
          n(2) = LocalNodes(4)
          n(3) = LocalNodes(1)

          Call match_list(Totele, Nloc, &
               D3, &
               ENList, &
               OppNode, n, &
               LocalElements, p, &
               ComEle, OtherNode)

          EEList(EEmark) = ComEle
          EEmark = EEmark + 1

          if(ComEle.eq.0) then
             ErrMark = ErrMark+1
          end if

          OppNode = LocalNodes(3)
          n(1) = LocalNodes(4)
          n(2) = LocalNodes(1)
          n(3) = LocalNodes(2)

          Call match_list(Totele, Nloc, &
               D3, &
               ENList, &
               OppNode, n, &
               LocalElements, p, &
               ComEle, OtherNode)

          EEList(EEmark) = ComEle
          EEmark = EEmark + 1

          if(ComEle.eq.0) then
             ErrMark = ErrMark+1
          end if

          OppNode = LocalNodes(4)
          n(1) = LocalNodes(1)
          n(2) = LocalNodes(2)
          n(3) = LocalNodes(3)

          Call match_list(Totele, Nloc, &
               D3, &
               ENList, &
               OppNode, n, &
               LocalElements, p, &
               ComEle, OtherNode)

          EEList(EEmark) = ComEle
          EEmark = EEmark + 1

          if(ComEle.eq.0) then
             ErrMark = ErrMark+1
          end if

          if(ErrMark.eq.4.and.totele/=1) then
             ewrite(-1,*) '...all surface eles zero..', ele
             FLAbort("dieing")
          end if

       Elseif (nloc==2) then
          ! 1D
          ! Note that nloc==2 is very unsafe but will do pending a complete
          ! rewrite of adjacency_lists.
          call MakeEEList1D(Nonods,Totele,NLoc,&
               ENList, &
               NEList, lgthNEList, NEListBasePtr,  &
               EEList)

       else
          ! now 2d

          OppNode = LocalNodes(1)
          n(1) = LocalNodes(2)
          n(2) = LocalNodes(3)

          Call match_list(Totele, Nloc, &
               D3, &
               ENList, &
               OppNode, n, &
               LocalElements, p, &
               ComEle, OtherNode)

          EEList(EEmark) = ComEle
          EEmark = EEmark + 1

          if(ComEle.eq.0) then
             ErrMark = ErrMark+1
          end if

          OppNode = LocalNodes(2)
          n(1) = LocalNodes(3)
          n(2) = LocalNodes(1)

          Call match_list(Totele, Nloc, &
               D3, &
               ENList, &
               OppNode, n, &
               LocalElements, p, &
               ComEle, OtherNode)

          EEList(EEmark) = ComEle
          EEmark = EEmark + 1

          if(ComEle.eq.0) then
             ErrMark = ErrMark+1
          end if

          OppNode = LocalNodes(3)
          n(1) = LocalNodes(1)
          n(2) = LocalNodes(2)

          Call match_list(Totele, Nloc, &
               D3, &
               ENList, &
               OppNode, n, &
               LocalElements, p, &
               ComEle, OtherNode)

          EEList(EEmark) = ComEle
          EEmark = EEmark + 1

          if(ComEle.eq.0) then
             ErrMark = ErrMark+1
          end if

          if(ErrMark.eq.4.and.totele/=1 ) then
             ewrite(3,*) '...all sur eles zero..', ele
             stop 4445
          end if

       End if

    End Do

    Deallocate(LocalElements, LocalNodes, n)

    ewrite(3, *) "At end of subroutine MakeEEList_old"

    Return
  End Subroutine MakeEEList_old

  Subroutine MakeEEList1D(Nonods,Totele,NLoc,&
       ENList, &
       NEList, lgthNEList, NEListBasePtr,  &
       EEList)

    !! This subroutine returns ordered list of elements connected to 'ele'
    !! Zero entry indicates a free boundary, ie. on the surface

    Implicit None

    Integer, Intent(In)::Nonods, Totele, NLoc
    Integer, Intent(In)::ENList(totele*nloc)
    Integer, intent(In)::lgthNEList, NEList(lgthNElist), NEListBasePtr(Nonods+1)

    Integer, Intent(Out)::EEList(Totele*NLoc)

    integer :: ele, ele2, node, i, j, EEMark
    integer, dimension(nloc) :: ele_nodes, ele2_nodes
    ! This is inefficient but it is assumed this will only be used for tests.
    integer, dimension(:), allocatable :: node_eles

    EEList=0

    do ele=1,totele
       ele_nodes=ENList((ele-1)*nloc+1:ele*nloc)

       do i=1,size(ele_nodes)
          node=ele_nodes(i)
          allocate(node_eles(NEListBasePtr(node+1)-NEListBasePtr(node)))
          node_eles=NEList(NEListBasePtr(node):NEListBasePtr(node+1)-1)

          do j=1, size(node_eles)
             ele2=node_eles(j)
             ele2_nodes=ENList((ele2-1)*nloc+1:ele2*nloc)

             if (ele/=ele2) then
                if (any(ele_nodes(1)==ele2_nodes)) then
                   EEMark=2
                else
                   EEMark=1
                end if
                EEList((ele-1)*nloc+EEMark)=ele2
             end if

          end do
          deallocate(node_eles)
       end do

    end do

  end Subroutine MakeEEList1D

  ! ****************************************************************

  Subroutine LocalElementsNods(Nonods, NLoc, &
       NEList, NEListBasePtr, lgthNEList, &
       LocalNodes, &
       biglgth, LocalElements, p)

    ! this subroutine gives list of all elements connected to the entries of LocalNodes


    Implicit None

    Integer, intent(in)::Nonods, NLoc
    Integer, Intent(In)::lgthNEList, NEListBasePtr(Nonods+1), NEList(lgthNEList)
    Integer, INtent(In)::LocalNodes(NLoc)

    Integer, Intent(In)::biglgth
    Integer, Intent(Out)::p, LocalElements(biglgth)
    Integer  i, iLoc, iNod,ptr

    ! local...

    Logical Inserted

    ! Find elements each of these nodes is connected to, and search these eles only..
    ! Need to remove the duplications..


    p=0

    Do iLoc=1,NLoc
       ! Loop over the local nodes..

       iNod = LocalNodes(iLoc)

       if(iNod.gt.Nonods) then
          ewrite(3,*) 'Node out of range'
          stop 886
       end if


       Do ptr = NEListBasePtr(iNod), NEListBasePtr(iNod+1) - 1

          If(p.eq.0) then
             p = p+1
             LocalElements(p) = NEList(ptr)
          Else

             Inserted = .False.

             do i=1,p
                if(LocalElements(i).eq.NEList(ptr)) then
                   inserted = .True.
                end if
             end do

             if(.not.Inserted) then
                p = p+1
                if(p.gt.biglgth) then
                   ewrite(3,*) 'p out of bounds',p,biglgth
                   stop 887
                end if
                LocalElements(p) = NEList(ptr)
             End IF
          End If
       End Do
    End Do

    Return
  end Subroutine LocalElementsNods

  ! *********************************************************

  Subroutine match_list(Totele, Nloc, &
       D3, &
       ndglno, &
       OppNode, n, &
       TestList, lgthTestList, &
       Element, OtherNode)

    Implicit None

    Integer, Intent(in)::Totele, NLoc, ndglno(totele*nloc)
    Logical, INtent(In)::D3
    Integer, Intent(In)::OppNode, n(NLoc-1)
    Integer, Intent(In)::lgthTestList, TestList(lgthTestList)
    Integer, Intent(Out)::Element, OtherNode
    ! Local...
    Integer ele, iptr
    Integer i
    Integer n1, n2, n3
    Logical FirstNode, SecondNode, ThirdNode, Matched

    Logical, Allocatable::UsedNodes(:)
    Integer, Allocatable::m(:)

    Allocate(UsedNodes(Nloc))
    Allocate(m(Nloc))

    If(D3) Then
       n1 = n(1)
       n2 = n(2)
       n3 = n(3)
       ! oppNode forms 4th (or 3rd in 2d) entry..
    Else
       n1 = n(1)
       n2 = n(2)
    End if


    Element = 0

    do iptr=1,lgthTestList
       ele = TestList(iptr)  ! this is element to try...

       ! put nodes of element being tried in array..
       do i=1,NLoc
          m(i) = ndglno((ele-1)*nloc+i)
          UsedNodes(i) = .False.
       end do

       FirstNode = .False.
       SecondNode = .False.
       ThirdNode = .False. ! only use for 3d
       Matched = .False.
       ! match three of the nodes of element

       ! orders and sorting... complete match

       ! first node...
       do i=1,NLoc
          if (n1.eq.m(i)) then
             !    ewrite(3,*) 'Match on first node:',i,n1,m(i)
             UsedNodes(i) = .True.
             FirstNode = .True.
             exit
          end if
       end do

       !if one node doesn't match then its not correct element..

       if(FirstNode) then  ! now look for a second one...

          do i=1,Nloc
             if ((n2.eq.m(i)).AND.(.not.UsedNodes(i))) then
                UsedNodes(i) = .True.
                SecondNode = .True.
                exit
             end if
          end do

       end if

       ! if 3D then look for 2nd and 3rd Node, 2D just a second node..

       if(D3) Then

          if(SecondNode) then  ! now look for a second one..., only go in here is firstnode= T

             do i=1,NLoc
                if ((n3.eq.m(i)).AND.(.not.UsedNodes(i))) then
                   UsedNodes(i) = .True.
                   ThirdNode = .True.
                   exit
                end if
             end do

          end if

          if(ThirdNode) then ! need choice of one node when reach this loop..

             do i=1,NLoc ! filter to find correct node..

                if(.not.(UsedNodes(i))) then

                   if (OppNode.eq.m(i)) then
                      ! all 4 nodes in.. found the element
                      !            ewrite(3,*) 'element - but original one:',l,ele
                      ! if equal then continue as in current element..
                      Element = 0
                   else
                      ! this is different so is the correct answer..
                      !           ewrite(3,*) 'element - new:',l,ele
                      Element = ele
                      OtherNode = m(i)
                      Matched = .True.
                      exit ! does this work?
                   end if
                end if
             end do

          end if

       Else

          if(SecondNode) then ! need choice of one node when reach this loop..

             do i=1,NLoc ! filter to find correct node..

                if(.not.(UsedNodes(i))) then

                   if (OppNode.eq.m(i)) then
                      ! all 4 nodes in.. found the element
                      !  ewrite(3,*) 'element - but original one:',i,ele
                      ! if equal then continue as in current element..
                      Element = 0
                   else
                      ! this is different so is the correct answer..
                      !           ewrite(3,*) 'element - new:',l,ele
                      Element = ele
                      OtherNode = m(i)
                      Matched = .True.
                      exit ! does this work?
                   end if
                end if
             end do

          end if

       End IF

       ! ewrite(3,*) 'matched:',Matched
       If(Matched) exit

    end do

    !  ewrite(3,*) 'Final matched element:',ele

    !print *,'Nodes are n:',n1,n2,n3, OppNode  ! these two lists should be different..
    !print *,'Nodes are m:',m
    ! ewrite(3,*) 'OtherNode:',OtherNode
    !  ewrite(3,*) '****************************************'

    Deallocate(UsedNodes, m)

    return
  End Subroutine match_list

  ! ***************************************************************
  subroutine MakeLists_Dynamic(Nonods, Totele, Nloc, NDGLNO, D3, NEList,&
       & NNList, EEList)
    ! Dynamic version of Makelists which returns csr_sparsitys as adjacency
    ! graphs. This version of MakeLists only returns the arrays
    ! you ask for.
    !
    Integer, Intent(In)::nonods, Totele, NLoc
    Integer, Intent(In)::NdGlNo(Totele*NLoc)
    Logical, Intent(In)::D3

    type(csr_sparsity), intent(out), optional :: NEList, NNList, EEList

    ! We need an NEList even if we're not making one.
    type(csr_sparsity) :: lNEList
    integer :: i
    integer :: lgthNEList
    Logical ISSUE

    ! Note that the NEList is needed to calculate EEList.
    if (present(NEList).or.present(EEList)) then

       ! Form node to element list +++++++++++++++++++++

       ! Allocate provably enough space.
       call allocate(lNEList, rows=nonods, columns=totele, &
            entries=totele*nloc, diag=.false., name='NEListSparsity')

       CALL NODELE(NONODS,lNElist%findrm,lNEList%colm,&
            lgthNEList, totele*nloc,&
            TOTELE,NLOC,NDGLNO)

       !ASSERT(lNEList%findrm(nonods+1)==lgthnelist+1)
    end if

    if (present(EEList)) then

       ! ++++++++++++++++++++++++++++++++++++++++++++
       ! Form E-E List
       call allocate(EEList, rows=totele, columns=totele, &
            entries=totele*nloc, diag=.false., name='EEListSparsity')


       Call MakeEEList_old(Nonods,Totele,NLoc,&
            D3, ndglno,&
            lNEList%colm, lgthNEList, lNEList%findrm, &
            EEList%colm)

       forall (i=1:totele+1)
          EEList%findrm(i)=nloc*(i-1)+1
       end forall


    end if

    ! Use or deallocate the temporary node:element list.
    if (present(NEList)) then
      NEList=lNEList
    else if (present(EEList)) then
      call deallocate(lNEList)
    end if

    if (present(NNList)) then

       ! +++++++++++++++++++++++++++++++++++++++++
       !Form NN List

       ! This uses the dynamically allocated version from the sparse_tools
       ! module instead if the partially static MakeNNList.
       call POSINM(NNList, TOTELE, nonods, nloc, ndglno,&
            nonods, nloc, ndglno, diag=.false., name='NNListSparsity')

    end if

    ! ++++++++++++++++++++++++++++++++++++++++++
    ! do some checking..

    ISSUE = .false.

    if (present(NNList)) then
       if(any(NNList%colm.gt.Nonods)) then
          ewrite(3,*) 'problem with NNlist'
          ISSUE = .true.
       end if
    end if

    if (present(NEList)) then
       if (any(NEList%colm.gt.Totele)) then
          ewrite(3,*) 'problem with NElist'
          ISSUE = .true.
       end if
    end if

    if (present(EEList)) then
       if(any(EEList%colm.gt.Totele)) then
          ewrite(3,*) 'problem with EElist'
          ISSUE = .true.
       end if
    end if

    if(ISSUE) then
       stop 96788
    end if

  end subroutine MakeLists_Dynamic

  subroutine MakeLists_Mesh(mesh, NEList, NNList, EEList)
    !!< Use the new mesh type.
    type(mesh_type), intent(in) :: mesh
    type(csr_sparsity), intent(out), optional :: NEList, NNList, EEList

    type(csr_sparsity) lNEList
    integer :: nonods, totele, nloc
    integer, dimension(:), pointer :: ndglno
    logical :: d3

    nonods = mesh%nodes
    totele = mesh%elements
    nloc   = mesh%shape%ndof
    ndglno => mesh%ndglno
    d3     = (mesh%shape%dim == 3)

    if (mesh%continuity < 0) then
      ewrite(0,*) "Warning: asking for adjacency lists of discontinuous mesh"
      if (present(EElist)) then
        FLAbort("and you asked for the eelist.")
      end if
    end if

    if (present(NEList) .and. present(NNList) .and. present(EEList)) then
      call MakeLists_Dynamic(Nonods, Totele, Nloc, NDGLNO, D3, NEList=NEList,&
       & NNList=NNList, EEList=EEList)
      return
    end if

    if (present(NEList) .and. present(NNList)) then
      call MakeLists_Dynamic(Nonods, Totele, Nloc, NDGLNO, D3, NEList=NEList,&
       & NNList=NNList)
      return
    end if

    if (present(EEList)) then
      ! we need to also construct at least NEList
      if (present(NNList)) then
        ! also construct NNList
        call MakeLists_Dynamic(Nonods, Totele, Nloc, NDGLNO, D3, &
        & NNList=NNList, NEList=lNEList)
      else
        call MakeLists_Dynamic(Nonods, Totele, Nloc, NDGLNO, D3, NEList=lNEList)
      end if
      ewrite(1,*) 'Using the new MakeEElist'
      call MakeEEList(EEList, mesh, lNEList)
      if (present(NEList)) then
        NEList=lNEList
      else
        call deallocate(lNEList)
      end if
      return
    end if

    if (present(NEList)) then
      call MakeLists_Dynamic(Nonods, Totele, Nloc, NDGLNO, D3, NEList=NEList)
      return
    end if

    if (present(NNList)) then
      call MakeLists_Dynamic(Nonods, Totele, Nloc, NDGLNO, D3, NNList=NNList)
      return
    end if

    if (present(EEList)) then
      call MakeLists_Dynamic(Nonods, Totele, Nloc, NDGLNO, D3, EEList=EEList)
      return
    end if

  end subroutine MakeLists_Mesh

  subroutine MakeEEList(EEList, mesh, NEList)
  !!< For a given mesh and Node-Element list calculate the
  !!< Element-Element list.

    type(csr_sparsity), intent(out):: EEList
    type(mesh_type), intent(in):: mesh
    type(csr_sparsity), intent(in):: NEList

    integer, dimension(:), pointer:: cols
    integer adj_ele, ele, noboundaries, nloc, j
#ifdef DDEBUG
    integer, dimension(:), allocatable :: debug_common_elements
    integer :: no_found
#endif

    ! Number of element boundaries.
    noboundaries=facet_count(mesh%shape)
    if (mesh%elements<=0) then
      call allocate(EEList, rows=0, columns=0, entries=0, name='EEListSparsity')
      return
    end if
    nloc=size(mesh%ndglno)/mesh%elements

    call allocate(EEList, rows=mesh%elements, columns=mesh%elements, &
         entries=mesh%elements*noboundaries, name='EEListSparsity')

    EEList%findrm=(/  (1+ele*noboundaries, ele=0,  mesh%elements) /)

    do ele=1, mesh%elements
       cols => row_m_ptr(EEList, ele)
       assert(size(cols) == noboundaries)
       do j=1, noboundaries
          ! fill in element on the other side of face j:
          call find_adjacent_element(ele, adj_ele, NEList, &
               nodes=mesh%ndglno((ele-1)*nloc+ &
               facet_dofs(mesh%shape, j) &
               )  )
#ifdef DDEBUG
          if(adj_ele >= 0) then
#endif
             ! Found an adjacent element, or no adjacent elements (a boundary)
             cols(j) = adj_ele
#ifdef DDEBUG
          else
             ! Encountered an error
             ewrite(-1, *) "For element ", ele, " with facet ", mesh%ndglno((ele - 1) * nloc + &
               & facet_dofs(mesh%shape, j))
             allocate(debug_common_elements(0))
             call findcommonelements(debug_common_elements, no_found, nelist, &
               & nodes = mesh%ndglno((ele - 1) * nloc + &
               & facet_dofs(mesh%shape, j) &
               & ))
             ewrite(-1, *) "Number of common elements: ", no_found
             deallocate(debug_common_elements)
             allocate(debug_common_elements(no_found))
             call findcommonelements(debug_common_elements, no_found, nelist, &
               & nodes = mesh%ndglno((ele - 1) * nloc + &
               & facet_dofs(mesh%shape, j) &
               & ))
             ewrite(-1, *) "Common elements: ", debug_common_elements
             deallocate(debug_common_elements)
             FLExit("Invalid NEList! Something wrong with the mesh?")
          end if
#endif
       end do

    end do

  contains

    subroutine find_adjacent_element(ele, adj_ele, nelist, nodes)
      !!< Using nelist find an element adjacent to ele with boundary nodes
      !!< nodes. Returns negative adj_ele on error. Error checking requires a
      !!< debugging build.

      !! The element for which we are seeking a neighbour
      integer, intent(in) :: ele
      !! The neighbour found
      integer, intent(out) :: adj_ele
      !! Node-Element list
      type(csr_sparsity), intent(in) :: nelist
      !! Boundary nodes on element ele
      integer, dimension(:), intent(in) :: nodes

      integer, dimension(:), pointer :: elements1
      integer :: i, j, candidate_ele

      ! Use an uninitialised integer_vector type, as the null initialisations
      ! have a cost and are not required here
      type uninit_integer_vector
        integer, dimension(:), pointer :: ptr
      end type uninit_integer_vector
      type(uninit_integer_vector), dimension(size(nodes) - 1) :: row_idx

      ! All elements connected to node nodes(1). One of these will be ele, and
      ! (if the boundary nodes are not on the domain boundary) one will be the
      ! adjacent element.
      elements1 => row_m_ptr(nelist, nodes(1))
      ! Cache the elements connected to nodes(2:). ele and (if the facet
      ! nodes are not on the domain boundary) the adjacent element will appear
      ! in all of these.
      do i = 2, size(nodes)
        row_idx(i - 1)%ptr => row_m_ptr(nelist, nodes(i))
      end do

      adj_ele = 0
      ele_loop: do i = 1, size(elements1)
         candidate_ele = elements1(i)
         if(candidate_ele == ele) cycle ele_loop  ! Ignore the query element
         ! See if this element borders all other nodes
         do j = 2, size(nodes)
           ! If candidate_ele is not in all row_idx, nodes are not facet
           ! nodes for candidate_ele, and it isn't the adjacent element.
           if(.not. any(row_idx(j - 1)%ptr == candidate_ele)) cycle ele_loop
         end do

#ifndef DDEBUG
         if(adj_ele > 0) then
           ! We've found more than one adjacent element. We're in trouble.
           adj_ele = -1
           return
         end if
#endif
         adj_ele = candidate_ele
#ifndef DDEBUG
         ! We've found the adjacent element. We're done.
         return
#endif
      end do ele_loop

    end subroutine find_adjacent_element

  end subroutine MakeEEList

  subroutine FindCommonElements(elements, n, NEList, nodes)
    !!< Using NEList find the elements that border all given nodes
    !! The elements found, and their number n:
    integer, dimension(:), intent(out):: elements
    !! NOTE: The caller of this routine *has to*
    !! check that the returned number of elements n <= size(elements)
    !!  As I can't give a useful error message here
    !!  about what (presumably) is wrong with the mesh
    integer, intent(out):: n
    !! Node-Element list:
    type(csr_sparsity), intent(in):: NEList
    !! The 'given' nodes:
    integer, dimension(:), intent(in):: nodes

    integer, dimension(:), pointer:: elements1
    integer i, j, ele

    type(integer_vector), dimension(size(nodes)-1) :: row_idx

    ! we'll loop over all elements bordering nodes(1)
    elements1 => row_m_ptr( NEList, nodes(1) )

    do j=2,size(nodes)
      row_idx(j-1)%ptr => row_m_ptr(NEList, nodes(j))
    end do

    n=0
    ele_loop: do i=1, size(elements1)
       ele=elements1(i)
       ! see if this element borders all other nodes
       do j=2, size(nodes)
          if (.not. any( row_idx(j-1)%ptr==ele )) cycle ele_loop
       end do

       ! if so we have found a common element
       n=n+1
       if (n<=size(elements)) elements(n)=ele

    end do ele_loop

  end subroutine FindCommonElements

  function make_edge_list(topology) result (edge_list)
    ! Given a topology mesh (ie linear, continuous), return the set of
    !  edges in that mesh.

    type(mesh_type), intent(in) :: topology
    type(csr_sparsity) :: edge_list

    type(integer_set), dimension(:), allocatable :: edge_sets
    type(cell_type), pointer :: cell
    integer, dimension(2) :: edge_vertices
    integer, dimension(:), pointer :: ele_vertices, edge_row
    integer :: ele, edge, vertex

    assert(topology%shape%degree==1)

    allocate(edge_sets(node_count(topology)))
    call allocate(edge_sets)

    cell=>topology%shape%cell
    do ele=1,element_count(topology)
       ele_vertices => ele_nodes(topology, ele)

       do edge=1, cell%entity_counts(1)
          edge_vertices=ele_vertices(entity_vertices(cell,[1,edge]))

          call insert(edge_sets(edge_vertices(1)), edge_vertices(2))
          call insert(edge_sets(edge_vertices(2)), edge_vertices(1))

       end do
    end do

    call allocate(edge_list, rows=node_count(topology), &
         columns=node_count(topology), entries=sum(key_count(edge_sets)), &
         name="EdgeList")

    edge_list%findrm=1
    do vertex=1,node_count(topology)
       edge_list%findrm(vertex+1)=&
            edge_list%findrm(vertex)+key_count(edge_sets(vertex))

       edge_row=>row_m_ptr(edge_list,vertex)

       edge_row=set2vector(edge_sets(vertex))
    end do

    call deallocate(edge_sets)
    deallocate(edge_sets)

  end function make_edge_list

end module adjacency_lists






