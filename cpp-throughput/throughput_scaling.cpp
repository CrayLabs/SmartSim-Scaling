#include "client.h"
#include "logger.h"
#include "srexception.h"
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

    //Initializing rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //Indicate Client creation
    if(!rank)
        log_data(context, LLInfo, "Connecting clients");
        std::cout<<"Connecting clients"<<std::endl<<std::flush;

    //Creating client
    double constructor_start = MPI_Wtime();
    bool cluster = get_cluster_flag();
    SmartRedis::Client client(cluster, context);
    double constructor_end = MPI_Wtime();

    //Storing client() time
    double delta_t = constructor_end - constructor_start;
    timing_file << rank << "," << "client()" << ","
                << delta_t << "\n";
    std::string text4 = "client() time stored for rank: ";
    text4 += std::to_string(rank);
    log_data(context, LLInfo, text4);

    // Block to make sure all clients are connected
    MPI_Barrier(MPI_COMM_WORLD);

    //Andrew
    size_t n_values = n_bytes / sizeof(float);
    std::vector<float> array(n_values, 0);
    std::vector<float> result(n_values, 0);
    for(size_t i=0; i<n_values; i++)
        array[i] = i;


    std::vector<double> put_tensor_times;
    std::vector<double> unpack_tensor_times;

    //Grabbing iterations 
    int iterations = get_iterations();
    std::string text = "Running with iterations: ";
    text += std::to_string(iterations);
    log_data(context, LLDebug, text);

    //Waiting for every MPI rank
    MPI_Barrier(MPI_COMM_WORLD);

    // Keys are overwritten in order to help
    // ensure that the database does not run out of memory
    // for large messages.
    log_data(context, LLDebug, "Iteration loop starting...");
    double loop_start = MPI_Wtime();
    for (int i=0; i<iterations; i++) {
        std::string text1 = "Running iteration: ";
        text1 += std::to_string(i);
        log_data(context, LLDebug, text1);

        std::string key = "throughput_rank_" +
                          std::to_string(rank);

        log_data(context, LLDebug, "put_tensor started");
        double put_tensor_start = MPI_Wtime();
        client.put_tensor(key,
                          array.data(),
                          {1, n_values},
                          SRTensorTypeFloat, 
                          SRMemLayoutContiguous);
        double put_tensor_end = MPI_Wtime();
        log_data(context, LLDebug, "put_tensor ended");
        delta_t = put_tensor_end - put_tensor_start;
        put_tensor_times.push_back(delta_t);

        log_data(context, LLDebug, "unpack_tensor started");
        double unpack_tensor_start = MPI_Wtime();
        client.unpack_tensor(key,
                             result.data(),
                             {n_values},
                             SRTensorTypeFloat,
                             SRMemLayoutContiguous);
        double unpack_tensor_end = MPI_Wtime();
        log_data(context, LLDebug, "unpack_tensor ended");
        delta_t = unpack_tensor_end - unpack_tensor_start;
        unpack_tensor_times.push_back(delta_t);

        std::string text3 = "Ending iteration: ";
        text3 += std::to_string(i);
        log_data(context, LLDebug, text3);
    }
    double loop_end = MPI_Wtime();
    log_data(context, LLDebug, "All iterations complete");
    log_data(context, LLDebug, "Writing data to files");
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

    //Waiting for every MPI rank
    MPI_Barrier(MPI_COMM_WORLD);
    log_data(context, LLDebug, "Data written to files");
    return;
}

int main(int argc, char* argv[]) {

    std::string context("Scaling Tests");
    MPI_Init(&argc, &argv);

    //initializing rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //Bill & Andrew
    if(argc==1)
        // log_error(context, LLInfo, "The number tensor size in "\
        //                          "bytes must be provided as "\
        //                          "a command line argument.");
        throw std::runtime_error("The number tensor size in "\
                                 "bytes must be provided as "\
                                 "a command line argument.");

    //Andrew
    std::string s_bytes(argv[1]);
    int n_bytes = std::stoi(s_bytes);

    //Throughput test starts
    log_data(context, LLDebug, "Starting Throughput test");
    double main_start = MPI_Wtime();

    //Indicating tensor size used in throughput test
    std::string text = "Running throughput scaling test with tensor size of ";
    text += std::to_string(n_bytes);
    text += " bytes.";
    if(rank==0)
        log_data(context, LLInfo, text);
        std::cout<<"Running throughput scaling test with tensor size of "
                 <<n_bytes<<"bytes."<<std::endl;

    //Opening timing file
    std::ofstream timing_file;
    timing_file.open("rank_" + std::to_string(rank) + "_timing.csv");

    //Calling Throughput method
    run_throughput(timing_file, n_bytes);

    //Ending Throughput test
    double main_end = MPI_Wtime();

    //Indicate test end to user
    if(rank==0)
        log_data(context, LLInfo, "Finished throughput test.");
        std::cout<<"Finished throughput test."<<std::endl;

    //Logging total Throughput time to file
    double delta_t = main_end - main_start;
    timing_file << rank << "," << "main()" << ","
                << delta_t << std::endl << std::flush;

    MPI_Finalize();

    return 0;
}
