program main

use utils, only : get_env_var
use smartredis_client, only : client_type
use mpi_f08

implicit none

! Configuration parameters
integer :: batch_size, num_devices, client_count
character(len=255) :: device_type
logical :: should_set, use_cluster

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
should_set = get_env_var("SS_SET_MODEL", .false.)
use_cluster = get_env_var("SS_CLUSTER", .false.)
client_count = get_env_var("SS_CLIENT_COUNT", 18)

call MPI_init(ierror)
main_start = MPI_Wtime()

call MPI_comm_rank(MPI_COMM_WORLD, rank, ierror)
write(timing_file,'(A,I0,A)') 'rank_', rank, '_timing.csv'
open(newunit = timing_unit, &
    file = timing_file,     &
    status = 'REPLACE'      &
    )

call init_client(client, rank, use_cluster, timing_unit)
if (should_set .and. rank == 0) call set_model(client, device_type, num_devices, batch_size)
write(model_key,'(A,I0)') 'resnet_model_',mod(rank,num_devices)
write(script_key,'(A,I0)') 'resnet_script_',mod(rank,num_devices)

if (rank==0) write(*,*) "Finished Resnet test"
main_end = MPI_Wtime()
delta_t = main_end - main_start

write(timing_unit,'(I0,A,G0)') rank, ',main(),', delta_t

call MPI_finalize()

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

subroutine set_model(client, device_type, num_devices, batch_size)
    type(client_type), intent(in) :: client
    character(len=*),  intent(in) :: device_type
    integer,           intent(in) :: num_devices
    integer,           intent(in) :: batch_size

    integer :: i
    integer :: return_code
    character(len=255) :: model_filename, script_filename
    character(len=255) :: model_key, script_key

    write(model_filename,'(A,A,A)') "./resnet50.", TRIM(device_type), '.pt'
    write(script_filename,'(A,A,A)') "./dataprocessing_script.txt.", TRIM(device_type), '.pt'

    if (num_devices > 1 .and. device_type == 'GPU') then
        model_key = 'resnet_model_0'
        script_key = 'resnet_script_0'
        return_code = client%set_model_from_file_multigpu( &
            model_key, model_filename, "TORCH", 0, num_devices, batch_size)
        return_code = client%set_script_from_file_multigpu(script_key, script_filename, 0, num_devices)
    else
        do i=1,num_devices
            write(model_key,'(A,I0)') 'resnet_model_',i
            write(script_key,'(A,I0)') 'resnet_script_',i
            return_code = client%set_model_from_file(model_key, model_filename, "TORCH", device_type, batch_size)
            return_code = client%set_script_from_file(script_key, device_type, script_filename)
        enddo
    endif
end subroutine set_model

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
    real(kind=4), dimension(224,224,3) :: array
    real(kind=8), dimension(100) :: put_tensor_times
    real(kind=8), dimension(100) :: run_script_times
    real(kind=8), dimension(100) :: run_model_times
    real(kind=8), dimension(100) :: unpack_tensor_times

    character(len=255) :: in_key, script_out_key, out_key
    logical :: use_multigpu

    integer :: i, return_code

    use_multigpu = (num_devices > 0) .and. (device_type == 'GPU')
    call random_number(array)
    call MPI_Barrier(MPI_COMM_WORLD)

    write(in_key,'(A,I0,A,I0)') 'resnet_input_rank_', rank, '_', i
    write(script_out_key,'(A,I0,A,I0)') 'resnet_processed_input_rank_', rank, '_', i
    write(out_key,'(A,I0,A,I0)') 'resnet_output_rank_', rank, '_', i

    do i=1,100
        put_tensor_start = MPI_WTime()
        return_code = client%put_tensor(in_key, array, [224, 224,3])
        put_tensor_end = MPI_WTime()
        if (return_code/=SRNoError) stop "Error in put tensor"
        put_tensor_times(i) = put_tensor_end-put_tensor_start

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
        if (return_code/=SRNoError) stop "Error in put tensor"
        run_script_times(i) = run_script_end-run_script_start
    enddo


end subroutine run_mnist
end program main