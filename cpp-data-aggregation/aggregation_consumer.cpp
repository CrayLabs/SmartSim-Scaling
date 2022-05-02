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

    int n_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_rank);

    if (rank == 0)
        std::cout << "Connecting clients" << std::endl;

    double constructor_start = MPI_Wtime();
    SmartRedis::Client client(true);
    double constructor_end = MPI_Wtime();
    double delta_t = constructor_end - constructor_start;
    timing_file << rank << "," << "client()" << ","
                << delta_t << "\n";

    // allocate arrays to hold timings
    std::vector<double> get_list_times;

    int iterations = get_iterations();

    MPI_Barrier(MPI_COMM_WORLD);

    double loop_start = MPI_Wtime();

    for (int i=0; i<iterations; i++) {

        std::string list_name = "iteration_" + std::to_string(i);

        std::cout << "Consuming list " << i << std::endl;

        if(rank == 0) {
            bool list_is_ready = client.poll_list_length(list_name, list_length,
                                                         5, 100000);
            if(!list_is_ready)
                throw std::runtime_error("There was an error in the "\
                                         "aggregation scaling test.  "\
                                         "The list never reached size of " +
                                         std::to_string(list_length));
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double get_list_start = MPI_Wtime();
        std::vector<SmartRedis::DataSet> result =
            client.get_datasets_from_list(list_name);
        double get_list_end = MPI_Wtime();
        delta_t = get_list_end - get_list_start;
        get_list_times.push_back(delta_t);

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            client.delete_list(list_name);
        }
    }

    double loop_end = MPI_Wtime();
    delta_t = loop_end - loop_start;

    // write times to file
    for (int i=0; i<iterations; i++) {
        timing_file << rank << "," << "get_list" << ","
                    << get_list_times[i] << "\n";
    }

    timing_file << rank << "," << "loop_time" << ","
                << delta_t << "\n";

    timing_file << std::flush;


    return;
}

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double main_start = MPI_Wtime();

    if(argc==1)
        throw std::runtime_error("The expected list length must be "
                                 "passed in.");

    std::string s_bytes(argv[1]);
    int n_bytes = std::stoi(s_bytes);


    int list_length = 0;

    if(rank==0)
        std::cout << "Running aggregate scaling test consumer." << std::endl;

    //Open Timing file
    std::ofstream timing_file;
    timing_file.open("rank_" + std::to_string(rank) + "_timing.csv");

    run_aggregation_consumer(timing_file, list_length);

    if(rank==0)
        std::cout << "Finished aggregation scaling consumer." << std::endl;

    double main_end = MPI_Wtime();
    double delta_t = main_end - main_start;
    timing_file << rank << "," << "main()" << ","
                << delta_t << std::endl << std::flush;

    MPI_Finalize();

    return 0;
}
