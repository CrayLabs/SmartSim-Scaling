#include "client.h"
#include <mpi.h>


int get_iterations() {
  char* iterations = std::getenv("SS_ITERATIONS");
  int iters = iterations ? std::stoi(iterations) : 100;
  return iters;
}

void run_aggregation_consumer(std::ofstream& timing_file,
                              int list_length)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
        std::cout << "Connecting clients" << std::endl;

    // Connect the client and save connection time
    double constructor_start = MPI_Wtime();
    SmartRedis::Client client(true);
    double constructor_end = MPI_Wtime();
    double delta_t = constructor_end - constructor_start;
    timing_file << rank << "," << "client()" << ","
                << delta_t << "\n";

    // Allocate arrays to hold timings
    std::vector<double> get_list_times;

    // Retrieve the number of iterations to run
    int iterations = get_iterations();

    // Block to make sure all clients are connected
    MPI_Barrier(MPI_COMM_WORLD);

    // Retrieve rank-local loop start time
    double loop_start = MPI_Wtime();

    // Perform dataset aggregation retrieval
    for (int i=0; i<iterations; i++) {

        // Create aggregation list name
        std::string list_name = "iteration_" + std::to_string(i);

        if (rank == 0) {
            std::cout << "Consuming list " << i << std::endl;
        }

        // Have rank 0 check that the aggregation list is full
        if(rank == 0) {
            bool list_is_ready = client.poll_list_length(list_name,
                                                         list_length,
                                                         5, 100000);
            if(!list_is_ready)
                throw std::runtime_error("There was an error in the "\
                                         "aggregation scaling test.  "\
                                         "The list never reached size of " +
                                         std::to_string(list_length));
        }

        // Have all ranks wait until the aggregation list is full
        MPI_Barrier(MPI_COMM_WORLD);

        // Have each rank retrieve the datasets in the aggregation list
        double get_list_start = MPI_Wtime();
        std::vector<SmartRedis::DataSet> result =
            client.get_datasets_from_list(list_name);
        double get_list_end = MPI_Wtime();
        delta_t = get_list_end - get_list_start;
        get_list_times.push_back(delta_t);

        // Block until all ranks are complete with aggregation
        MPI_Barrier(MPI_COMM_WORLD);

        // Delete the list so the producer knows the list has been consumed
        if (rank == 0) {
            client.delete_list(list_name);
        }
    }

    // Compute loop execution time
    double loop_end = MPI_Wtime();
    delta_t = loop_end - loop_start;

    // Write aggregation times to file
    for (int i = 0; i < iterations; i++) {
        timing_file << rank << "," << "get_list" << ","
                    << get_list_times[i] << "\n";
    }

    // Write loop time to file
    timing_file << rank << "," << "loop_time" << ","
                << delta_t << "\n";

    // Flush the output stream
    timing_file << std::flush;

    return;
}

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double main_start = MPI_Wtime();

    // Get command line arguments
    if(argc==1)
        throw std::runtime_error("The expected list length must be "
                                 "passed in.");

    std::string s_list_length(argv[1]);
    int list_length = std::stoi(s_list_length);

    if(rank==0)
        std::cout << "Running aggregate scaling test consumer." << std::endl;

    // Open Timing file
    std::ofstream timing_file;
    timing_file.open("rank_" + std::to_string(rank) + "_timing.csv");

    // Run the aggregation scaling study
    run_aggregation_consumer(timing_file, list_length);

    if(rank==0)
        std::cout << "Finished aggregation scaling consumer." << std::endl;

    // Save time it took to run main function
    double main_end = MPI_Wtime();
    double delta_t = main_end - main_start;
    timing_file << rank << "," << "main()" << ","
                << delta_t << std::endl << std::flush;

    MPI_Finalize();

    return 0;
}
