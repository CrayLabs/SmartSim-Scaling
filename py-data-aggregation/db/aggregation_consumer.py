from __future__ import annotations

import os
import sys
from mpi4py import MPI

import typing as t

from smartredis import Client


def get_iterations() -> int:
    try:
        return int(os.getenv("SS_ITERATIONS", "20"))
    except ValueError:
        return 20


def run_aggergation_consumer(timing_file: t.TextIO, list_length: int) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("Connecting clients")

    constructor_start = MPI.Wtime()
    client = Client(cluster=True)
    constructor_stop = MPI.Wtime()
    delta_t = constructor_stop - constructor_start
    timing_file.write(f"{rank},client(),{delta_t}\n")

    get_list_times: list[float] = []
    iterations = get_iterations()

    comm.Barrier()

    loop_start: float = MPI.Wtime()

    for i in range(iterations):
        list_name = f"iteration_{i}"
        if rank == 0:
            print(f"Consuming list {i}")

        if rank == 0:
            list_is_ready: bool = client.poll_list_length(
                name=list_name,
                list_length=list_length,
                poll_frequency_ms=5,
                num_tries=100000,
            )
            if not list_is_ready:
                raise RuntimeError(
                    "There was an error in the aggregation scaling test.  "
                    f"The list never reached size of {list_length}"
                )

        comm.Barrier()

        get_list_start = MPI.Wtime()
        _result = client.get_datasets_from_list(list_name)
        get_list_end = MPI.Wtime()
        delta_t = get_list_end - get_list_start
        get_list_times.append(delta_t)

        comm.Barrier()

        if rank == 0:
            client.delete_list(list_name)

    loop_end = MPI.Wtime()
    delta_t = loop_end - loop_start

    for time in get_list_times:
        timing_file.write(f"{rank},get_list,{time}\n")

    timing_file.write(f"{rank},loop_time,{delta_t}\n")
    timing_file.flush()


def main() -> int:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    main_start = MPI.Wtime()

    if len(sys.argv) == 1:
        raise RuntimeError("The expected list length must be passed in")
    list_length = int(sys.argv[1])

    with open(f"rank_{rank}_timing.csv", "w") as timing_file:
        run_aggergation_consumer(timing_file, list_length)
        if rank == 0:
            print("Finished aggregation scaling consumer.\n")

        main_end = MPI.Wtime()
        delta_t = main_end - main_start
        timing_file.write(f"{rank},main(),{delta_t}\n")
        timing_file.flush()

    MPI.Finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
