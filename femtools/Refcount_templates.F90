  subroutine addref_REFCOUNT_TYPE(object)
    !!< Increment the reference count of object creating a new reference
    !!< counter if needed.
    use parallel_tools, only: abort_if_in_parallel_region
    type(REFCOUNT_TYPE), intent(inout), target :: object
    integer, save :: id = 0

    call abort_if_in_parallel_region
    if (associated(object%refcount)) then
       ! Reference count already exists, just increment it.
       object%refcount%count=object%refcount%count+1

    else
       id = id + 1
       object%refcount=>new_refcount("REFCOUNT_TYPE", object%name)
       object%refcount%id = id
    end if

  end subroutine addref_REFCOUNT_TYPE

  subroutine incref_REFCOUNT_TYPE(object)
    !!< Increment the reference count of object. If there are no references
    !!< then error.
    use parallel_tools, only: abort_if_in_parallel_region
    type(REFCOUNT_TYPE), intent(in), target :: object
    integer, pointer :: ptr !! Dummy pointer to evade compilers which
    !! don't understand the rules for intent.

    call abort_if_in_parallel_region
    if (.not.associated(object%refcount)) then
       FLAbort ("Attempt to incref REFCOUNT_TYPE "//trim(object%name)//" which has no references")
    end if

    ! Reference count already exists, just increment it.
    ptr=>object%refcount%count
    ptr=ptr+1

  end subroutine incref_REFCOUNT_TYPE

  subroutine decref_REFCOUNT_TYPE(object)
    !!< Decrement the reference count on object. If the reference count drops
    !!< to 0 deallocate the refcount as a hint to the calling routine that
    !!< the object can safely be deallocated.
    use parallel_tools, only: abort_if_in_parallel_region
    type(REFCOUNT_TYPE), intent(inout) :: object

    call abort_if_in_parallel_region
    if (.not.associated(object%refcount)) then
       ! No refcount. Just exit
       return
    end if

    object%refcount%count=object%refcount%count-1

    if (object%refcount%count<=0) then

       if (object%refcount%count<0) then
          ! Warn for negative reference count
          ewrite(0,'(a, i0)') "Reference count of &
               &REFCOUNT_TYPE "//trim(object%name)//&
               " is ", object%refcount%count
          FLAbort("that should never happen.")
       end if

       object%refcount%prev%next=>object%refcount%next
       if (associated(object%refcount%next)) then
          object%refcount%next%prev=>object%refcount%prev
       end if

       deallocate(object%refcount)

    end if

  end subroutine decref_REFCOUNT_TYPE

  pure function has_references_REFCOUNT_TYPE(object) result (has_references)
    !!< Return true if there are any references to object
    type(REFCOUNT_TYPE), intent(in) :: object
    logical :: has_references

    has_references=associated(object%refcount)

  end function has_references_REFCOUNT_TYPE

