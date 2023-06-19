#include "client.h"
#include <mpi.h>


int get_iterations() {
  char* iterations = std::getenv("SS_ITERATIONS");
  int iters = iterations ? std::stoi(iterations) : 100;
  return iters;
}

bool get_cluster_flag() {
  char* cluster_flag = std::getenv("SS_CLUSTER");
  bool use_cluster = cluster_flag ? std::stoi(cluster_flag) : false;
  return use_cluster;
}

void run_throughput(std::ofstream& timing_file,
                    size_t n_bytes)
{
    std::string context("Run Throughput");
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    log_data(context, LLDebug, "rank: " + std::to_string(rank) + " initiated");

    if(!rank)
        log_data(context, LLInfo, "Connecting clients");
        std::cout<<"Connecting clients"<<std::endl<<std::flush;

    double constructor_start = MPI_Wtime();
    bool cluster = get_cluster_flag();
    SmartRedis::Client client(cluster, context);
    double constructor_end = MPI_Wtime();
    double delta_t = constructor_end - constructor_start;
    timing_file << rank << "," << "client()" << ","
                << delta_t << "\n";

    //print rank # storing client() for debugging
    std::string client_text = "client() time stored for rank: ";
    client_text += std::to_string(rank);
    log_data(context, LLDebug, client_text);

    MPI_Barrier(MPI_COMM_WORLD);
    size_t n_values = n_bytes / sizeof(float);
    std::vector<float> array(n_values, 0);
    std::vector<float> result(n_values, 0);
    for(size_t i=0; i<n_values; i++)
        array[i] = i;


    std::vector<double> put_tensor_times;
    std::vector<double> unpack_tensor_times;

    int iterations = get_iterations();

    MPI_Barrier(MPI_COMM_WORLD);

    //print cluster flag and rank for debug, signal start of iteration loop
    log_data(context, LLDebug, "Starting iterations with flags: cluster = " + std::to_string(cluster) + " for rank = " + std::to_string(rank));

    // Keys are overwritten in order to help
    // ensure that the database does not run out of memory
    // for large messages.
    double loop_start = MPI_Wtime();


    for (int i=0; i<iterations; i++) {
        //print out iteration # and rank # for debugging
        std::string iteration_debug = "Running iteration: ";
        iteration_debug += std::to_string(i);
        iteration_debug += " of rank: " + rank;
        log_data(context, LLDebug, iteration_debug);

        std::string key = "throughput_rank_" +
                          std::to_string(rank);

        double put_tensor_start = MPI_Wtime();
        client.put_tensor(key,
                          array.data(),
                          {1, n_values},
                          SRTensorTypeFloat, 
                          SRMemLayoutContiguous);
        double put_tensor_end = MPI_Wtime();
        delta_t = put_tensor_end - put_tensor_start;
        put_tensor_times.push_back(delta_t);
        log_data(context, LLDebug, "put_tensor completed for iteration: " + std::to_string(i) + "rank: " + std::to_string(rank));

        double unpack_tensor_start = MPI_Wtime();
        client.unpack_tensor(key,
                             result.data(),
                             {n_values},
                             SRTensorTypeFloat,
                             SRMemLayoutContiguous);
        double unpack_tensor_end = MPI_Wtime();
        delta_t = unpack_tensor_end - unpack_tensor_start;
        unpack_tensor_times.push_back(delta_t);
        log_data(context, LLDebug, "unpack_tensor completed for iteration: " + std::to_string(i) + "rank: " + std::to_string(rank));
    }

    double loop_end = MPI_Wtime();
    delta_t = loop_end - loop_start;

    // write times to file
    for (int i=0; i<iterations; i++) {

        timing_file << rank << "," << "put_tensor" << ","
                    << put_tensor_times[i] << "\n";

        timing_file << rank << "," << "unpack_tensor" << ","
                    << unpack_tensor_times[i] << "\n";

    }
    timing_file << rank << "," << "loop_time" << ","
                << delta_t << "\n";

    timing_file << std::flush;
    MPI_Barrier(MPI_COMM_WORLD);
    log_data(context, LLDebug, "run_throughput() method completed and all tensor times written to file");
    return;
}

int main(int argc, char* argv[]) {
    std::string context("Throughput Scaling Tests");
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    log_data(context, LLDebug, "rank: " + std::to_string(rank) + " initiated");

    if(argc==1)
        throw std::runtime_error("The number tensor size in "\
                                 "bytes must be provided as "\
                                 "a command line argument.");

    std::string s_bytes(argv[1]);
    int n_bytes = std::stoi(s_bytes);

    log_data(context, LLDebug, "Starting Throughput tests for rank: " + std::to_string(rank));
    double main_start = MPI_Wtime();

    //Indicating tensor size used in throughput test
    std::string tensor_text = "Running throughput scaling test with tensor size of ";
    tensor_text += std::to_string(n_bytes);
    tensor_text += " bytes.";
    if(rank==0)
        log_data(context, LLInfo, tensor_text);
        std::cout<<"Running throughput scaling test with tensor size of "
                 <<n_bytes<<"bytes."<<std::endl;


    std::ofstream timing_file;
    timing_file.open("rank_" + std::to_string(rank) + "_timing.csv");

    run_throughput(timing_file, n_bytes);

    if(rank==0)
        log_data(context, LLInfo, "Finished throughput test.");
        std::cout<<"Finished throughput test."<<std::endl;

    double main_end = MPI_Wtime();
    double delta_t = main_end - main_start;
    timing_file << rank << "," << "main()" << ","
                << delta_t << std::endl << std::flush;
    //print rank # storing client() for debugging
    std::string main_text = "main() time stored for rank: ";
    main_text += std::to_string(rank);
    log_data(context, LLDebug, main_text);

    MPI_Finalize();
    log_data(context, LLDebug, "All ranks finalized");

    return 0;
}