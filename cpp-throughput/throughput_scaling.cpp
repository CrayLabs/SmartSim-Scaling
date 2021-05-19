#include "client.h"
#include <mpi.h>

void run_throughput(std::ofstream& timing_file,
                    size_t n_bytes)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(!rank)
        std::cout<<"Connecting clients"<<std::endl<<std::flush;

    double constructor_start = MPI_Wtime();
    SmartRedis::Client client(true);
    double constructor_end = MPI_Wtime();
    double delta_t = constructor_end - constructor_start;
    timing_file << rank << "," << "client()" << ","
                << delta_t << std::endl << std::flush;

    MPI_Barrier(MPI_COMM_WORLD);

    size_t n_values = n_bytes / sizeof(float);
    std::vector<float> array(n_values, 0);
    std::vector<float> result(n_values, 0);
    for(size_t i=0; i<n_values; i++)
        array[i] = i;

    MPI_Barrier(MPI_COMM_WORLD);

    double loop_start = MPI_Wtime();

    // Keys are overwritten in order to help
    // ensure that the database does not run out of memory
    // for large messages.
    for (int i=0; i<10; i++) {

        std::string key = "throughput_rank_" +
                          std::to_string(rank);

        double put_tensor_start = MPI_Wtime();
        client.put_tensor(key, array.data(), {1, n_values},
                          SmartRedis::TensorType::flt,
                          SmartRedis::MemoryLayout::contiguous);
        double put_tensor_end = MPI_Wtime();
        delta_t = put_tensor_end - put_tensor_start;
        timing_file << rank << "," << "put_tensor" << ","
                    << delta_t << std::endl << std::flush;

        double unpack_tensor_start = MPI_Wtime();
        client.unpack_tensor(key, result.data(), {n_values},
                            SmartRedis::TensorType::flt,
                            SmartRedis::MemoryLayout::contiguous);
        double unpack_tensor_end = MPI_Wtime();
        delta_t = unpack_tensor_end - unpack_tensor_start;
        timing_file << rank << "," << "unpack_tensor" << ","
                    << delta_t << std::endl << std::flush;
    }

    double loop_end = MPI_Wtime();
    delta_t = loop_end - loop_start;
    timing_file << rank << "," << "loop_time" << ","
                << delta_t << std::endl << std::flush;

    return;
}

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(argc==1)
        throw std::runtime_error("The number tensor size in "\
                                 "bytes must be provided as "\
                                 "a command line argument.");

    std::string s_bytes(argv[1]);
    int n_bytes = std::stoi(s_bytes);

    double main_start = MPI_Wtime();

    if(rank==0)
        std::cout<<"Running throughput scaling test with tensor size of "
                 <<n_bytes<<"bytes."<<std::endl;


    //Open Timing file
    std::ofstream timing_file;
    timing_file.open("rank_" + std::to_string(rank) + "_timing.csv");

    run_throughput(timing_file, n_bytes);

    if(rank==0)
        std::cout<<"Finished throughput test."<<std::endl;

    double main_end = MPI_Wtime();
    double delta_t = main_end - main_start;
    timing_file << rank << "," << "main()" << ","
                << delta_t << std::endl << std::flush;

    MPI_Finalize();

    return 0;
}
