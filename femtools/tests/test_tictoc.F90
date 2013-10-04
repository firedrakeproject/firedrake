subroutine test_tictoc

  use tictoc
  use unittest_tools

  implicit none

  real :: current_cpu_time, start_cpu_time

  call tictoc_reset()
  call tic(TICTOC_ID_SIMULATION)

  call cpu_time(start_cpu_time)
  current_cpu_time = start_cpu_time
  do while(current_cpu_time - start_cpu_time < 0.1)
    call cpu_time(current_cpu_time)
  end do

  call toc(TICTOC_ID_SIMULATION)

  call report_test("[tictoc]", tictoc_time(TICTOC_ID_SIMULATION) < epsilon(0.0), .false., "Tictoc did not time")

end subroutine test_tictoc
