program main

use utils, only : get_env_var
use smartredis_client, only : client_type
use smartredis_errors, only : print_last_error
use mpi
use, intrinsic :: iso_fortran_env, only: error_unit

implicit none

! Configuration parameters
integer :: batch_size, num_devices, client_count
integer, parameter :: NUM_TRIALS = 100
character(len=255) :: device_type
logical :: use_cluster
logical :: poll_model_bool
logical :: poll_script_bool
integer :: poll_model_code
integer :: poll_script_code

! File imports
include "enum_fortran.inc"

! MPI-related variables
integer :: rank, ierror

! Timing variables
real(kind=8) :: main_start, main_end
real(kind=8) :: client_start, client_end
real(kind=8) :: delta_t
character(len=255) :: timing_file

! I/O Variables
integer :: timing_unit

! SmartRedis variables
type(client_type) :: client
character(len=255) :: script_key, model_key

! Retrieve all environment variables
batch_size = get_env_var("SS_BATCH_SIZE", 1)
device_type = get_env_var("SS_DEVICE", "GPU")
num_devices = get_env_var("SS_NUM_DEVICES", 1)
use_cluster = get_env_var("SS_CLUSTER", .false.)
client_count = get_env_var("SS_CLIENT_COUNT", 18)

! Initialize MPI and start the clock for timing the entire application
call MPI_init(ierror)
main_start = MPI_Wtime()
call MPI_comm_rank(MPI_COMM_WORLD, rank, ierror)

! Open the timing file that this rank will write to
write(timing_file,'(A,I0,A)') 'rank_', rank, '_timing.csv'
open(newunit = timing_unit, &
    file = timing_file,     &
    status = 'REPLACE'      &
    )

! Initialize the SmartRedis client
call init_client(client, rank, use_cluster, timing_unit)

! Check to that the ML model is in the database (loaded in the driver script)
model_key = "resnet_model"
poll_model_code = client%poll_model(model_key, 200, 100, poll_model_bool)
if (poll_model_code /= SRNoError) stop 'Something went wrong during poll_model execution'
if (.not. poll_model_bool) stop 'Model was not found'

! Check to that the processing script is in the database (loaded in the driver script)
script_key = "resnet_script"
poll_script_code = client%poll_model(script_key, 200, 100, poll_script_bool)
if (poll_script_code /= SRNoError) stop 'Something went wrong during poll_model execution'
if (.not. poll_script_bool) stop 'Script was not found'

call MPI_Barrier(MPI_COMM_WORLD, ierror)
! Run the main loop that runs the MNIST model
call run_mnist(rank, num_devices, device_type, model_key, script_key, timing_unit)

! Finalize the program
if (rank==0) write(*,*) "Finished Resnet test"
main_end = MPI_Wtime()
delta_t = main_end - main_start

! Write out the timing for the entire program
write(timing_unit,'(I0,A,G0)') rank, ',main(),', delta_t

call MPI_finalize(ierror)

contains

subroutine init_client( client, rank, use_cluster, timing_unit )
    type(client_type), intent(inout) :: client
    integer,           intent(in   ) :: rank
    logical,           intent(in   ) :: use_cluster
    integer,           intent(in   ) :: timing_unit
    integer :: return_code
    include "enum_fortran.inc"

    client_start = MPI_Wtime()
    return_code = client%initialize(use_cluster)
    client_end = MPI_Wtime()
    if (return_code /= SRNoError) stop 'Error initializing client'
    write(timing_unit,'(I0,A,G0)') rank, ',client(),', delta_t

end subroutine init_client

subroutine run_mnist(rank, num_devices, device_type, model_key, script_key, timing_unit)
    integer,          intent(in   ) :: rank
    integer,          intent(in   ) :: num_devices
    character(len=*), intent(in   ) :: device_type
    character(len=*), intent(in   ) :: model_key
    character(len=*), intent(in   ) :: script_key
    integer,          intent(in   ) :: timing_unit

    include "enum_fortran.inc"

    real(kind=8) :: delta_t
    real(kind=8) :: construct_start, construct_end
    real(kind=8) :: put_tensor_start, put_tensor_end
    real(kind=8) :: run_script_start, run_script_end
    real(kind=8) :: run_model_start, run_model_end
    real(kind=8) :: unpack_tensor_start, unpack_tensor_end
    real(kind=8) :: loop_start, loop_end
    real(kind=8), dimension(NUM_TRIALS) :: put_tensor_times
    real(kind=8), dimension(NUM_TRIALS) :: run_script_times
    real(kind=8), dimension(NUM_TRIALS) :: run_model_times
    real(kind=8), dimension(NUM_TRIALS) :: unpack_tensor_times

    real(kind=4), dimension(224,224,3) :: array
    real(kind=4), dimension(1,1000) :: result

    character(len=255) :: in_key, script_out_key, out_key
    logical :: use_multigpu

    integer :: i, return_code

    use_multigpu = (num_devices > 1) .and. (device_type == 'GPU')
    call random_number(array)

    ! Create the keys to use for this rank
    write(in_key,'(A,I0,A,I0)') 'resnet_input_rank_', rank, '_',1
    write(script_out_key,'(A,I0,A,I0)') 'resnet_processed_input_rank_', rank, '_', 1
    write(out_key,'(A,I0,A,I0)') 'resnet_output_rank_', rank, '_', 1

    ! Warmup the database before starting the loop
    return_code = client%put_tensor(in_key, array, [224, 224,3])
    if (return_code/=SRNoError) stop "Error in put tensor"

    if (use_multigpu) then
        return_code = client%run_script_multigpu( &
            script_key, "pre_process_3ch", [in_key], [script_out_key], rank, 0, num_devices)
        write(error_unit, *) 'is multi 1'
    else
        return_code = client%run_script(trim(script_key), "pre_process_3ch", [in_key], [script_out_key])
        write(error_unit, *) 'is not multi 1', script_key
    endif
    if (return_code /= SRNoError) then
        call print_last_error()
        stop "Error in run_script (warmup)"
    endif

    if (use_multigpu) then
        return_code = client%run_model_multigpu(model_key, [script_out_key], [out_key], rank, 0, num_devices)
    else
        return_code = client%run_model(model_key, [script_out_key], [out_key])
    endif
    if (return_code /= SRNoError) then
        call print_last_error()
        stop "Error in run_model"
    endif

    return_code = client%unpack_tensor(out_key, result, [1,1000])
    if (return_code/=SRNoError) stop "Error in put tensor"

    ! Start the main loop
    loop_start = MPI_WTime()
    do i=1, NUM_TRIALS
        ! Ensure that all ranks try to run an inference at the same time
        call MPI_Barrier(MPI_COMM_WORLD, ierror)

        write(in_key,'(A,I0,A,I0)') 'resnet_input_rank_', rank, '_', i
        write(script_out_key,'(A,I0,A,I0)') 'resnet_processed_input_rank_', rank, '_', i
        write(out_key,'(A,I0,A,I0)') 'resnet_output_rank_', rank, '_', i

        ! Put the tensor into the database
        put_tensor_start = MPI_WTime()
        return_code = client%put_tensor(in_key, array, [224, 224,3])
        put_tensor_end = MPI_WTime()
        if (return_code/=SRNoError) stop "Error in put tensor"
        put_tensor_times(i) = put_tensor_end-put_tensor_start

        ! Run the preprocessing script
        if (use_multigpu) then
            run_script_start = MPI_WTime()
            return_code = client%run_script_multigpu( &
                script_key, "pre_process_3ch", [in_key], [script_out_key], rank, 0, num_devices)
            run_script_end = MPI_WTime()
        else
            run_script_start = MPI_WTime()
            return_code = client%run_script(script_key, "pre_process_3ch", [in_key], [script_out_key])
            run_script_end = MPI_WTime()
        endif
        if (return_code/=SRNoError) stop "Error in run_script"
        run_script_times(i) = run_script_end-run_script_start

        ! Do the inference
        if (use_multigpu) then
            run_model_start = MPI_WTime()
            return_code = client%run_model_multigpu(model_key, [script_out_key], [out_key], rank, 0, num_devices)
            run_model_end = MPI_WTime()
        else
            run_model_start = MPI_WTime()
            return_code = client%run_model(model_key, [script_out_key], [out_key])
            run_model_end = MPI_WTime()
        endif
        if (return_code/=SRNoError) stop "Error in run_model"
        run_model_times(i) = run_model_end-run_model_start

        ! Retrieve the result from the database
        unpack_tensor_start = MPI_WTime()
        return_code = client%unpack_tensor(out_key, result, [1,1000])
        unpack_tensor_end = MPI_WTime()
        if (return_code/=SRNoError) stop "Error in put tensor"
        unpack_tensor_times(i) = unpack_tensor_end-unpack_tensor_start
    enddo
    loop_end = MPI_WTime()

    ! Write out the results of the times
    do i=1, NUM_TRIALS
        write(timing_unit,'(I0,A,E25.16)') rank, ",put_tensor,", put_tensor_times(i)
        write(timing_unit,'(I0,A,E25.16)') rank, ",run_script,", run_script_times(i)
        write(timing_unit,'(I0,A,E25.16)') rank, ",run_model,", run_model_times(i)
        write(timing_unit,'(I0,A,E25.16)') rank, ",unpack_tensor,", unpack_tensor_times(i)
    enddo


end subroutine run_mnist
end program main
