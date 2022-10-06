from __future__ import annotations

import os
import struct
import sys
import shutil
import time
import itertools
import numpy as np
import multiprocessing as mp
from mpi4py import MPI
import typing as t


# New datatypes
TDataSet = tuple[str, dict[str, np.ndarray]]

# Consts
SIZEOF_FLOAT: t.Final[int] = 4
SIZEOF_SIZE_T: t.Final[int] = 8


def get_iterations() -> int:
    try:
        return int(os.getenv("SS_ITERATIONS", "20"))
    except ValueError:
        return 20


def get_read_from_dir() -> str:
    read_from_dir = os.getenv("READ_FROM_DIR")
    if not read_from_dir:
        raise RuntimeError("Do not know from where to read incoming data")
    return read_from_dir


def poll_list_length(
    name: str, length: int, poll_frequency_ms: int, num_tries: int
) -> bool:
    list_path = os.path.join(get_read_from_dir(), name)
    while not os.path.exists(list_path) or len(os.listdir(list_path)) < length:
        if num_tries:
            num_tries -= 1
            time.sleep(poll_frequency_ms / 1000)
        else:
            return False
    return True


def get_datasets_from_list(list_name: str) -> list[TDataSet]:
    if sr_thread_count := os.getenv("SR_THREAD_COUNT"):
        num_workers = int(sr_thread_count)
    else:
        raise RuntimeError("env var SR_THREAD_COUNT is not set")
    list_path = os.path.join(get_read_from_dir(), list_name)
    dataset_files = [os.path.join(list_path, file) for file in os.listdir(list_path)]
    chunked_dataset_files = [dataset_files[i::num_workers] for i in range(num_workers)]

    with mp.Pool(num_workers) as pool:
        return list(
            itertools.chain.from_iterable(
                pool.map(_read_datasets_from_files, chunked_dataset_files)
            )
        )


# Keep top level in case of ``mp.set_start_method("spawn")``
def _read_datasets_from_files(filenames: list[str]) -> list[TDataSet]:
    datasets: list[TDataSet] = []
    for filename in filenames:
        with open(filename, "rb") as file:
            next_ = file.read(SIZEOF_SIZE_T)
            dataset_name = file.read(int.from_bytes(next_, "little")).decode("utf-8")
            dataset = {}
            while next_ := file.read(SIZEOF_SIZE_T):
                tensor_name = file.read(int.from_bytes(next_, "little")).decode("utf-8")
                next_ = file.read(SIZEOF_SIZE_T)
                tensor_data_b = file.read(
                    int.from_bytes(next_, "little") * SIZEOF_FLOAT
                )
                tensor_data = np.array(
                    struct.unpack(
                        "f" * (len(tensor_data_b) // SIZEOF_FLOAT), tensor_data_b
                    )
                )
                dataset[tensor_name] = tensor_data
            datasets.append((dataset_name, dataset))
    assert len(filenames) == len(datasets)
    return datasets


def run_aggergation_consumer(timing_file: t.TextIO, list_length: int) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("Connecting clients")

    delta_t = float(0)
    timing_file.write(f"{rank},client(),{delta_t}\n")

    get_list_times: list[float] = []
    iterations = get_iterations()

    comm.Barrier()

    loop_start = MPI.Wtime()

    for i in range(iterations):
        list_name = f"iteration_{i}"
        if rank == 0:
            print(f"Consuming list {i}")

        if rank == 0:
            list_is_ready = poll_list_length(
                name=list_name,
                length=list_length,
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
        _result = get_datasets_from_list(list_name)
        get_list_end = MPI.Wtime()
        delta_t = get_list_end - get_list_start
        get_list_times.append(delta_t)

        comm.Barrier()

        if rank == 0:
            shutil.rmtree(os.path.join(get_read_from_dir(), list_name))

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
