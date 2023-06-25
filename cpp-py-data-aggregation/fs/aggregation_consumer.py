from __future__ import annotations

import os
import io
import sys
import struct
import shutil
import time
import numpy as np
from multiprocessing.pool import ThreadPool
from mpi4py import MPI
import typing as t

from smartsim.log import get_logger, log_to_file

if t.TYPE_CHECKING:
    import numpy.typing as npt


# Consts
SIZEOF_FLOAT: t.Final[int] = 4
SIZEOF_SIZE_T: t.Final[int] = 8


def get_iterations() -> int:
    try:
        return int(os.getenv("SS_ITERATIONS", "20"))
    except ValueError:
        return 20


def get_read_from_dir() -> str:
    """Get the path to a directory to write datasets files 
    and make list dirs

    :return: path to dir
    :rtype: str
    """
    read_from_dir = os.getenv("READ_FROM_DIR")
    if not read_from_dir:
        raise RuntimeError("Do not know from where to read incoming data")
    return read_from_dir


def poll_list_length(
    name: str, length: int, poll_frequency_ms: int, num_tries: int
) -> bool:
    """A file system equivelent of ``smartredis.Client.poll_list_length``
    
    :param name: name of a dataset list
    :type name: str
    :param length: poll until list length surpases this val
    :type length: int
    :name poll_frequency_ms: number of millsec to wait between polls
    :type poll_frequency_ms: int
    :name num_tries: number of time to try polling the list
    :type num_tries: int
    :return: contents of the datasets
    :rtype: list[bytes]
    """
    list_path = os.path.join(get_read_from_dir(), name)
    while not os.path.exists(list_path) or len(os.listdir(list_path)) < length:
        if num_tries > 0:
            num_tries -= 1
            time.sleep(poll_frequency_ms / 1000)
        else:
            return False
    return True


def get_datasets_from_list(list_name: str) -> list[bytes]:
    """A file system equivelent of ``smartredis.Client.get_dataset_from_list``
    
    :param list_name: name of a dataset list
    :type list_name: str
    :return: contents of the datasets
    :rtype: list[bytes]
    """
    try:
        num_workers = int(os.getenv("SR_THREAD_COUNT", "4"))
    except ValueError:
        num_workers = 4
    list_path = os.path.join(get_read_from_dir(), list_name)
    dataset_files = [os.path.join(list_path, file) for file in os.listdir(list_path)]

    with ThreadPool(num_workers) as pool:
        return list(
            pool.imap_unordered(_read_dataset_from_file, dataset_files, chunksize=100)
        )


# Keep top level in case of ``mp.set_start_method("spawn")``
def _read_dataset_from_file(filename: str) -> bytes:
    """Read a dataset binary file from the filesystem
    
    :param filename: Path to a dataset binary file
    :type filename: str
    :return: file contents
    :rtype: bytes
    """
    with open(filename, "rb") as file:
        file_contents = file.read()
    return file_contents


def parse_dataset_bytes(
    dataset_bytes: bytes,
) -> tuple[str, dict[str, npt.NDArray[np.float64]]]:
    """A helper method to turn a string of bytes into an object
    equivilent to a smartredis.DataSet
    
    The resulting object is a tuple of lendgth 2 with:
        - a string at index 0 -> Dataset name
        - a dict [str, np.array] at index 1 -> Mapping of tensor name to tensor

    :param dataset_bytes: byte string read from filesystem
    :type dataset_bytes: bytes
    :returns: A dataset equivelent object
    :rtype: tuple[str, dict[str, npt.NDArray[np.float64]]]
    """
    with io.BytesIO(dataset_bytes) as file:
        next_ = file.read(SIZEOF_SIZE_T)
        dataset_name = file.read(int.from_bytes(next_, "little")).decode("utf-8")
        dataset: dict[str, npt.NDArray[np.float64]] = {}
        while next_ := file.read(SIZEOF_SIZE_T):
            tensor_name = file.read(int.from_bytes(next_, "little")).decode("utf-8")
            next_ = file.read(SIZEOF_SIZE_T)
            tensor_data_b = file.read(int.from_bytes(next_, "little") * SIZEOF_FLOAT)
            tensor_data = np.array(
                struct.unpack("f" * (len(tensor_data_b) // SIZEOF_FLOAT), tensor_data_b)
            )
            dataset[tensor_name] = tensor_data
    return dataset_name, dataset


def run_aggregation_consumer(timing_file: t.TextIO, list_length: int) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    logger = get_logger(f"Data Aggregation FS Consumer MPI Rank {rank}")
    logger.debug(f"Initialized Rank")

    if rank == 0:
        logger.info("Connecting clients")
        print("Connecting clients")

    # We will spend no time connecting a SR client bc we are not using SR
    delta_t = float(0)
    timing_file.write(f"{rank},client(),{delta_t}\n")
    logger.debug(f"client() time stored for rank: {rank}")

    # Allocate lists to hold timings
    get_list_times: list[float] = []
    poll_list_times: list[float] = []

    # Retrieve the number of iterations
    iterations = get_iterations()
    # std::string iteration_num = "Running with iterations: " + std::to_string(iterations);
    logger.debug(f"Running with iterations: {iterations}")

    # Block to make sure all clients connected
    comm.Barrier()

    # Retrive the rank-local loop start time
    logger.debug(f"loop_time() starting")
    loop_start = MPI.Wtime()
    
    # Preform dataset aggregation retrieval
    for i in range(iterations):
        logger.debug(f"Running iteration {i} of rank {rank}")
        
        # Create aggregation list name
        list_name = f"iteration_{i}"
        if rank == 0:
            print(f"Consuming list {i}")

        # Have rank 0 check that the aggregation list is full
        logger.debug("Starting poll_time()")
        poll_time_start = MPI.Wtime()
        if rank == 0:
            list_is_ready = poll_list_length(
                name=list_name,
                length=list_length,
                poll_frequency_ms=50,
                num_tries=100000,
            )
            if not list_is_ready:
                raise RuntimeError(
                    "There was an error in the aggregation scaling test.  "
                    f"The list never reached size of {list_length}"
                )

        # Have all ranks wait until the aggregation list is full
        comm.Barrier()
        poll_list_end = MPI.Wtime()
        logger.debug("Ended poll_time()")
        delta_t = poll_list_end - poll_time_start
        poll_list_times.append(delta_t)

        # Have each rank retrieve the datasets in the aggregation list
        logger.debug("Starting get_list()")
        get_list_start = MPI.Wtime()
        _result = get_datasets_from_list(list_name)
        get_list_end = MPI.Wtime()
        logger.debug("Ended get_list()")
        delta_t = get_list_end - get_list_start
        get_list_times.append(delta_t)

        # Block until all ranks are complete with aggregation
        comm.Barrier()

        # Delete the list so the producer knows the list has been consumed
        if rank == 0:
            shutil.rmtree(os.path.join(get_read_from_dir(), list_name))

    # Compute loop execution time
    loop_end = MPI.Wtime()
    logger.debug("Ended loop_time()")
    delta_t = loop_end - loop_start

    # Write aggregation times to file
    for write_time, read_time in zip(poll_list_times, get_list_times):
        timing_file.write(f"{rank},poll_list,{write_time}\n")
        timing_file.write(f"{rank},get_list,{read_time}\n")
    logger.debug("Data written to files")
    # Write loop times to file
    timing_file.write(f"{rank},loop_time,{delta_t}\n")

    # Flush the output stream
    timing_file.flush()


def main() -> int:
    #Initializing rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    logger = get_logger(f"Data Aggregation Tests Consumer Rank {rank}")
    logger.debug("Starting Data Aggregation Consumer fs test")
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
            logger.info("Finished aggregation scaling fs consumer.\n")
            print("Finished aggregation scaling fs consumer.\n")

        # Save time it took to run the main function
        main_end = MPI.Wtime()
        logger.debug("Ended Data Aggregation Consumer fs test")
        delta_t = main_end - main_start
        timing_file.write(f"{rank},main(),{delta_t}\n")
        timing_file.flush()

    MPI.Finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
