#include "client.h"
#include <mpi.h>


int get_iterations() {
  char* iterations = std::getenv("SS_ITERATIONS");
  int iters = iterations ? std::stoi(iterations) : 100;
  return iters;
}

void run_aggregation(std::ofstream& timing_file,
                     size_t n_bytes,
                     size_t tensors_per_dataset)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_rank);

    if(!rank)
        std::cout<<"Connecting clients"<<std::endl<<std::flush;

    double constructor_start = MPI_Wtime();
    SmartRedis::Client client(true);
    double constructor_end = MPI_Wtime();
    double delta_t = constructor_end - constructor_start;
    timing_file << rank << "," << "client()" << ","
                << delta_t << "\n";

    MPI_Barrier(MPI_COMM_WORLD);

    size_t n_values = n_bytes / sizeof(float);
    std::vector<float> array(n_values, 0);
    std::vector<float> result(n_values, 0);
    for(size_t i=0; i<n_values; i++)
        array[i] = i;


    // allocate arrays to hold timings
    std::vector<double> get_list_times;

    int iterations = get_iterations();

    MPI_Barrier(MPI_COMM_WORLD);

    double loop_start = MPI_Wtime();

    // Keys are overwritten in order to help
    // ensure that the database does not run out of memory
    // for large messages.
    for (int i=0; i<iterations; i++) {

        std::string list_name = "iteration_" + std::to_string(i);

        std::string name = "aggregation_rank_" +
                           std::to_string(rank);

        SmartRedis::DataSet dataset(name);
        for (size_t j = 0; j < tensors_per_dataset; j++) {
            std::string tensor_name = "tensor_" + std::to_string(j);
            dataset.add_tensor(tensor_name,
                               array.data(),
                               {1, n_values},
                               SRTensorTypeFloat,
                               SRMemLayoutContiguous);
        }

        client.put_dataset(dataset);
        client.append_to_list(list_name, dataset);

        if(rank == 0) {
            bool list_is_ready = client.poll_list_length(list_name, n_rank,
                                                         5, 100000);
            if(!list_is_ready)
                throw std::runtime_error("There was an error in the "\
                                         "aggregation scaling test.  "\
                                         "The list never reached size of " +
                                         std::to_string(n_rank));

            double get_list_start = MPI_Wtime();
            std::vector<SmartRedis::DataSet> result =
                client.get_datasets_from_list(list_name);
            double get_list_end = MPI_Wtime();
            delta_t = get_list_end - get_list_start;
            get_list_times.push_back(delta_t);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        // write times to file
        if(rank == 0) {
                timing_file << rank << "," << "get_list" << ","
                            << get_list_times.back() << "\n";
                timing_file << std::flush;
        }

    }

    double loop_end = MPI_Wtime();
    delta_t = loop_end - loop_start;
    /*
    // write times to file
    if(rank == 0) {
        for (int i=0; i<iterations; i++) {
            timing_file << rank << "," << "get_list" << ","
                        << get_list_times[i] << "\n";

        }
        timing_file << rank << "," << "loop_time" << ","
                    << delta_t << "\n";
    }
    timing_file << std::flush;
    */
    if (rank == 0) {
        timing_file << rank << "," << "loop_time" << ","
                    << delta_t << "\n";
        timing_file << std::flush;
    }
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

    if(argc==2)
        throw std::runtime_error("The number of tensors per "\
                                 "dataset must be provided as "\
                                 "a command line argument.");

    std::string s_bytes(argv[1]);
    int n_bytes = std::stoi(s_bytes);

    std::string s_tensors_per_dataset(argv[2]);
    int tensors_per_dataset = std::stoi(s_tensors_per_dataset);

    double main_start = MPI_Wtime();

    if(rank==0)
        std::cout<<"Running aggregate scaling test with tensor size of "
                 <<n_bytes<<" bytes and "<<tensors_per_dataset<<
                 " tensors per dataset."<<std::endl;


    //Open Timing file
    std::ofstream timing_file;
    timing_file.open("rank_" + std::to_string(rank) + "_timing.csv");

    run_aggregation(timing_file, n_bytes, tensors_per_dataset);

    if(rank==0)
        std::cout<<"Finished throughput test."<<std::endl;

    double main_end = MPI_Wtime();
    double delta_t = main_end - main_start;
    timing_file << rank << "," << "main()" << ","
                << delta_t << std::endl << std::flush;

    MPI_Finalize();

    return 0;
}
