module utils

implicit none; private

interface get_env_var
    module procedure get_env_var_char, get_env_var_int, get_env_var_logical
end interface get_env_var

public get_env_var

contains

function get_env_var_char(var_name, default_value) result(result)
    character(len=*), intent(in   ) :: var_name
    character(len=*), intent(in   ) :: default_value
    character(len=255) :: result

    character(len=255) :: interim_result

    integer :: status

    call get_environment_variable(var_name, interim_result, status = status)
    if (status /= 0) then
        result = default_value
    else
        result = interim_result
    endif

end function get_env_var_char

function get_env_var_int(var_name, default_value) result(result)
    character(len=*), intent(in   ) :: var_name
    integer,            intent(in   ) :: default_value
    integer :: result

    integer :: status
    character(len=255) :: interim_result

    call get_environment_variable(var_name, interim_result, status = status)
    if (status /= 0) then
        result = default_value
    else
        read(interim_result,*) result
    endif

end function get_env_var_int

function get_env_var_logical(var_name, default_value) result(result)
    character(len=*), intent(in   ) :: var_name
    logical,            intent(in   ) :: default_value
    logical :: result

    integer :: status, int_result
    character(len=255) :: interim_result

    call get_environment_variable(var_name, interim_result, status = status)
    if (status /= 0) then
        result = default_value
    else
        read(interim_result,*) int_result
        if (int_result == 0) then
            result = .false.
        else
            result = .true.
        endif
    endif

end function get_env_var_logical

end module utils