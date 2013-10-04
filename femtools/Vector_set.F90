module vector_set

  implicit none

  external :: vec_create_set, vec_clear_set, vec_is_present, vec_destroy_set
  external :: intvec_create_set, intvec_clear_set, intvec_is_present, intvec_destroy_set

  interface vecset_add
    module procedure vecset_is_present
  end interface

  interface intvecset_add
    module procedure intvecset_is_present
  end interface

  contains

  subroutine vecset_create(idx)
    integer, intent(out) :: idx
    call vec_create_set(idx)
  end subroutine vecset_create

  subroutine vecset_is_present(idx, v, bool)
    !!< Is v present in the set of vectors?
    !!< If not, add it.
    integer, intent(in) :: idx
    real, dimension(:), intent(in) :: v
    integer :: success
    logical, optional, intent(out) :: bool

    call vec_is_present(idx, v, size(v), success)
    if (present(bool)) then
      bool = (success == 0)
    end if
  end subroutine vecset_is_present

  subroutine vecset_clear(idx)
    !!< Clear the set.
    integer, intent(in) :: idx
    call vec_clear_set(idx)
  end subroutine vecset_clear

  subroutine vecset_destroy(idx)
    !!< Clear the set and deallocate if possible.
    integer, intent(in) :: idx
    call vec_destroy_set(idx)
  end subroutine vecset_destroy

  subroutine intvecset_create(idx)
    integer, intent(out) :: idx
    call intvec_create_set(idx)
  end subroutine intvecset_create

  subroutine intvecset_is_present(idx, v, bool)
    !!< Is v present in the set of intvectors?
    !!< If not, add it.
    integer, intent(in) :: idx
    integer, dimension(:), intent(in) :: v
    integer :: success
    logical, optional, intent(out) :: bool

    call intvec_is_present(idx, v, size(v), success)
    if (present(bool)) then
      bool = (success == 0)
    end if
  end subroutine intvecset_is_present

  subroutine intvecset_clear(idx)
    !!< Clear the set.
    integer, intent(in) :: idx
    call intvec_clear_set(idx)
  end subroutine intvecset_clear

  subroutine intvecset_destroy(idx)
    !!< Clear the set and deallocate if possible.
    integer, intent(in) :: idx
    call intvec_destroy_set(idx)
  end subroutine intvecset_destroy

end module vector_set
