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


def run_aggregation_consumer(timing_file: t.TextIO, list_length: int) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("Connecting clients")

    # Connect a client and save connection time
    constructor_start = MPI.Wtime()
    client = Client(cluster=True)
    constructor_stop = MPI.Wtime()
    delta_t = constructor_stop - constructor_start
    timing_file.write(f"{rank},client(),{delta_t}\n")

    # Allocate lists to hold timings
    poll_list_times: list[float] = []
    get_list_times: list[float] = []

    # Retrieve the number of iterations to run
    iterations = get_iterations()

    # Block to make sure all clients connected
    comm.Barrier()

    # Retrive the rank-local loop start time
    loop_start = MPI.Wtime()

    # Preform dataset aggregation retrieval
    for i in range(iterations):

        # Create aggregation list name
        list_name = f"iteration_{i}"

        if rank == 0:
            print(f"Consuming list {i}")
    
        # Have rank 0 check that the aggregation list is full
        poll_time_start = MPI.Wtime()
        if rank == 0:
            list_is_ready: bool = client.poll_list_length(
                name=list_name,
                list_length=list_length,
                poll_frequency_ms=50,
                num_tries=100_000,
            )
            if not list_is_ready:
                raise RuntimeError(
                    "There was an error in the aggregation scaling test.  "
                    f"The list never reached size of {list_length}"
                )

        # Have all ranks wait until the aggregation list is full
        comm.Barrier()
        poll_time_end = MPI.Wtime()
        delta_t = poll_time_end - poll_time_start
        poll_list_times.append(delta_t)

        # Have each rank retrieve the datasets in the aggregation list
        get_list_start = MPI.Wtime()
        _result = client.get_datasets_from_list(list_name)
        get_list_end = MPI.Wtime()
        delta_t = get_list_end - get_list_start
        get_list_times.append(delta_t)

        # Block until all ranks are complete with aggregation
        comm.Barrier()

        # Delete the list so the producer knows the list has been consumed
        if rank == 0:
            client.delete_list(list_name)

    # Compute loop execution time
    loop_end = MPI.Wtime()
    delta_t = loop_end - loop_start

    # Write aggregation times to file
    for poll_time, get_time in zip(poll_list_times, get_list_times):
        timing_file.write(f"{rank},poll_list,{poll_time}\n")
        timing_file.write(f"{rank},get_list,{get_time}\n")

    # Write loop times to file
    timing_file.write(f"{rank},loop_time,{delta_t}\n")

    # Flush the output stream
    timing_file.flush()


def main() -> int:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    main_start = MPI.Wtime()

    # Get command line arguments
    if len(sys.argv) == 1:
        raise RuntimeError("The expected list length must be passed in")
    list_length = int(sys.argv[1])

    # Open timing file
    with open(f"rank_{rank}_timing.csv", "w") as timing_file:
        # Run the aggregation scaling study
        run_aggregation_consumer(timing_file, list_length)
        if rank == 0:
            print("Finished aggregation scaling consumer.\n")

        # Save time it took to run the main function
        main_end = MPI.Wtime()
        delta_t = main_end - main_start
        timing_file.write(f"{rank},main(),{delta_t}\n")
        timing_file.flush()

    MPI.Finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
