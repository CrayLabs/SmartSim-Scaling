#include "client.h"
#include <mpi.h>


int get_iterations() {
  char* iterations = std::getenv("SS_ITERATIONS");
  int iters = iterations ? std::stoi(iterations) : 20;
  return iters;
}

void run_aggregation_consumer(std::ofstream& timing_file,
                              int list_length)
{
    //Initializing rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::string context("Data Aggregation MPI Consumer Rank: " + std::to_string(rank));
    log_data(context, LLDebug, "Initialized rank");

    //Indicate Client creation
    if (rank == 0)
        log_data(context, LLInfo, "Connecting clients");
        std::cout << "Connecting clients" << std::endl;

    // Connect the client and save connection time
    double constructor_start = MPI_Wtime();
    SmartRedis::Client client(true, context);
    double constructor_end = MPI_Wtime();
    double delta_t = constructor_end - constructor_start;
    timing_file << rank << "," << "client()" << ","
                << delta_t << "\n";
    //print rank # storing client() for debugging
    log_data(context, LLDebug, "client() time stored");

    // Allocate arrays to hold timings
    std::vector<double> get_list_times;

    // Allocate arrays to hold timings
    std::vector<double> poll_list_times;

    // Retrieve the number of iterations to run
    int iterations = get_iterations();
    log_data(context, LLDebug, "Running with iterations: " + std::to_string(iterations));

    // Block to make sure all clients are connected
    MPI_Barrier(MPI_COMM_WORLD);

    log_data(context, LLDebug, "Iteration loop starting...");

    // Retrieve rank-local loop start time
    double loop_start = MPI_Wtime();

    // Perform dataset aggregation retrieval
    for (int i=0; i<iterations; i++) {
        log_data(context, LLDebug, "Running iteration: " + std::to_string(i));

        // Create aggregation list name
        std::string list_name = "iteration_" + std::to_string(i);

        if (rank == 0) {
            std::cout << "Consuming list " << i << std::endl;
            log_data(context, LLInfo, "Consuming list " + std::to_string(i));
        }

        double poll_list_start = MPI_Wtime();
        // Have rank 0 check that the aggregation list is full
        if(rank == 0) {
            bool list_is_ready = client.poll_list_length(list_name,
                                                         list_length,
                                                         5, 100000);
            if(!list_is_ready) {
                std::string list_size_error = "There was an error in the "\
                                         "aggregation scaling test.  "\
                                         "The list never reached size of " +
                                         std::to_string(list_length);
                log_error(context, LLDebug, list_size_error);
                throw std::runtime_error(list_size_error);
            }
        }
        double poll_list_end = MPI_Wtime();
        log_data(context, LLDebug, "poll_list completed");
        delta_t = poll_list_end - poll_list_start;
        poll_list_times.push_back(delta_t);
        // Have all ranks wait until the aggregation list is full
        MPI_Barrier(MPI_COMM_WORLD);

        // Have each rank retrieve the datasets in the aggregation list
        double get_list_start = MPI_Wtime();
        std::vector<SmartRedis::DataSet> result =
            client.get_datasets_from_list(list_name);
        double get_list_end = MPI_Wtime();
        log_data(context, LLDebug, "get_list completed");
        delta_t = get_list_end - get_list_start;
        get_list_times.push_back(delta_t);

        // Block until all ranks are complete with aggregation
        MPI_Barrier(MPI_COMM_WORLD);

        // Delete the list so the producer knows the list has been consumed
        if (rank == 0) {
            client.delete_list(list_name);
            log_data(context, LLDebug, "Data Agg List " + list_name + " is deleted");
        }
    }
    // Compute loop execution time
    double loop_end = MPI_Wtime();
    log_data(context, LLDebug, "All iterations complete");
    delta_t = loop_end - loop_start;

    // Write aggregation times to file
    for (int i = 0; i < iterations; i++) {
        timing_file << rank << "," << "get_list" << ","
                    << get_list_times[i] << "\n";
        timing_file << rank << "," << "poll_list" << ","
                    << poll_list_times[i] << "\n";
    }

    // Write loop time to file
    timing_file << rank << "," << "loop_time" << ","
                << delta_t << "\n";

    log_data(context, LLDebug, "Data written to files");
    // Flush the output stream
    timing_file << std::flush;

    return;
}

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    //initializing rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string context("Data Aggregation Tests Consumer Rank: " + std::to_string(rank));
    log_data(context, LLDebug, "Rank initialized");
    log_data(context, LLDebug, "Starting Data Aggregation tests");
    double main_start = MPI_Wtime();

    // Get command line arguments
    if(argc==1) {
        std::string list_length_error = "The expected list length must be passed in.";
        log_error(context, LLInfo, list_length_error);
        throw std::runtime_error(list_length_error);
    }
    std::string s_list_length(argv[1]);
    int list_length = std::stoi(s_list_length);

    if(rank==0) {
        log_data(context, LLInfo, "Running aggregate scaling test consumer.");
        std::cout << "Running aggregate scaling test consumer." << std::endl;
    }
    // Open Timing file
    std::ofstream timing_file;
    timing_file.open("rank_" + std::to_string(rank) + "_timing.csv");

    // Run the aggregation scaling study
    run_aggregation_consumer(timing_file, list_length);

    // Save time it took to run main function
    double main_end = MPI_Wtime();

    //Indicate test end to user
    if(rank==0) {
        log_data(context, LLInfo, "Finished aggregation scaling consumer.");
        std::cout << "Finished aggregation scaling consumer." << std::endl;
    }
    //Logging total Data Agg time to file
    double delta_t = main_end - main_start;
    timing_file << rank << "," << "main()" << ","
                << delta_t << std::endl << std::flush;

    MPI_Finalize();

    return 0;
}