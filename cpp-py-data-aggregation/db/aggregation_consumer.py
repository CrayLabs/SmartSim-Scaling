from __future__ import annotations

import os
import sys
from mpi4py import MPI

import typing as t

from smartredis import Client

from smartsim.log import get_logger, log_to_file
logger = get_logger("Scaling Tests")


def get_iterations() -> int:
    try:
        return int(os.getenv("SS_ITERATIONS", "20"))
    except ValueError:
        return 20


def run_aggregation_consumer(timing_file: t.TextIO, list_length: int) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        logger.info("Connecting clients")
        print("Connecting clients")

    # Connect a client and save connection time
    constructor_start = MPI.Wtime()
    client = Client(cluster=True, logger)
    constructor_stop = MPI.Wtime()
    delta_t = constructor_stop - constructor_start
    timing_file.write(f"{rank},client(),{delta_t}\n")

    # Allocate lists to hold timings
    poll_list_times: list[float] = []
    get_list_times: list[float] = []

    # Retrieve the number of iterations to run
    iterations = get_iterations()
    logger.debug(f"Running with {iterations} iterations")

    # Block to make sure all clients connected
    comm.Barrier()

    # Retrive the rank-local loop start time
    logger.debug(f"loop_time() starting")
    loop_start = MPI.Wtime()

    # Preform dataset aggregation retrieval
    for i in range(iterations):
        logger.debug(f"Running iteration {i} out of {iterations}")

        # Create aggregation list name
        list_name = f"iteration_{i}"

        if rank == 0:
            print(f"Consuming list {i}")
            logger.info(f"Consuming list {i}")
    
        # Have rank 0 check that the aggregation list is full
        logger.debug("Starting poll_time()")
        poll_time_start = MPI.Wtime()
        if rank == 0:
            list_is_ready: bool = client.poll_list_length(
                name=list_name,
                list_length=list_length,
                poll_frequency_ms=50,
                num_tries=100_000,
            )
            if not list_is_ready:
                #Add
                raise RuntimeError(
                    "There was an error in the aggregation scaling test.  "
                    f"The list never reached size of {list_length}"
                )

        # Have all ranks wait until the aggregation list is full
        comm.Barrier()
        poll_time_end = MPI.Wtime()
        logger.debug("Ended poll_time()")
        delta_t = poll_time_end - poll_time_start
        poll_list_times.append(delta_t)

        # Have each rank retrieve the datasets in the aggregation list
        logger.debug("Starting get_list()")
        get_list_start = MPI.Wtime()
        _result = client.get_datasets_from_list(list_name)
        get_list_end = MPI.Wtime()
        logger.debug("Ended get_list()")
        delta_t = get_list_end - get_list_start
        get_list_times.append(delta_t)

        # Block until all ranks are complete with aggregation
        comm.Barrier()

        # Delete the list so the producer knows the list has been consumed
        if rank == 0:
            client.delete_list(list_name)

    # Compute loop execution time
    loop_end = MPI.Wtime()
    logger.debug("Ended loop_time()")
    delta_t = loop_end - loop_start

    # Write aggregation times to file
    for poll_time, get_time in zip(poll_list_times, get_list_times):
        timing_file.write(f"{rank},poll_list,{poll_time}\n")
        timing_file.write(f"{rank},get_list,{get_time}\n")
    logger.debug("Data written to files")
    # Write loop times to file
    timing_file.write(f"{rank},loop_time,{delta_t}\n")

    # Flush the output stream
    timing_file.flush()


def main() -> int:
    
    #Initializing rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    logger.debug("Starting Data Aggregation Consumer Py test")
    main_start = MPI.Wtime()

    # Get command line arguments
    if len(sys.argv) == 1:
        #Add
        raise RuntimeError("The expected list length must be passed in")
    list_length = int(sys.argv[1])

    # Open timing file
    with open(f"rank_{rank}_timing.csv", "w") as timing_file:
        # Run the aggregation scaling study
        run_aggregation_consumer(timing_file, list_length)
        if rank == 0:
            logger.info("Finished aggregation scaling py consumer.\n")
            print("Finished aggregation scaling py consumer.\n")

        # Save time it took to run the main function
        main_end = MPI.Wtime()
        logger.debug("Ended Data Aggregation Consumer Py test")
        delta_t = main_end - main_start
        timing_file.write(f"{rank},main(),{delta_t}\n")
        timing_file.flush()

    MPI.Finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
