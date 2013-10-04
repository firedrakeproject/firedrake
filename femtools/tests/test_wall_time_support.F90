subroutine test_wall_time_support

  use timers
  use unittest_tools

  implicit none

  call report_test("[Wall time supported]", .not. wall_time_supported(), .false., "Wall time not supported")

end subroutine test_wall_time_support
