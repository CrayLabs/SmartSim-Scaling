#include "client.h"
#include <mpi.h>


int get_iterations() {
  char* iterations = std::getenv("SS_ITERATIONS");
  int iters = iterations ? std::stoi(iterations) : 100;
  return iters;
}

void run_aggregation_production(size_t n_bytes,
                                size_t tensors_per_dataset)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_rank);

    if (rank == 0)
        std::cout << "Connecting clients" << std::endl;

    SmartRedis::Client client(true);

    MPI_Barrier(MPI_COMM_WORLD);

    size_t n_values = n_bytes / sizeof(float);
    std::vector<float> array(n_values, 0);
    std::vector<float> result(n_values, 0);
    for(size_t i = 0; i < n_values; i++)
        array[i] = i;

    int iterations = get_iterations();

    // Put the datasets into the database. The raw
    // dataset data can be re-used between iterations
    // because a new list is created for each iteration
    // We re-use datasets so we don't run out of memory.
    // Set the dataset name (MPI rank dependent)
    std::string name = "aggregation_rank_" +
                        std::to_string(rank);

    // Create the dataset and add specified number of tensors
    SmartRedis::DataSet dataset(name);
    for (size_t j = 0; j < tensors_per_dataset; j++) {
        std::string tensor_name = "tensor_" + std::to_string(j);
        dataset.add_tensor(tensor_name,
                            array.data(),
                            {1, n_values},
                            SRTensorTypeFloat,
                            SRMemLayoutContiguous);
    }

    // Put the dataset into the database
    client.put_dataset(dataset);

    // A new list is created for each iteration
    // to measure dataset aggregation throughput
    for (int i = 0; i < iterations; i++) {

        // Set the list name (not MPI rank dependent)
        std::string list_name = "iteration_" + std::to_string(i);

        // This poll invocation syncs the producer to wait for the
        // consumer to finish the previous iteration aggregation
        if (rank == 0 && i != 0) {
            std::string last_list_name = "iteration_" + std::to_string(i - 1);
            std::cout << "Checking that last list " << last_list_name << " is empty." << std::endl;
            while (client.get_list_length(last_list_name) != 0) {
                std::cout << "List length = " << client.get_list_length(last_list_name) << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "Creating list " << i << std::endl;
        }

        // Append to the aggregation list
        client.append_to_list(list_name, dataset);
    }
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

    if(rank==0)
        std::cout << "Running aggregate scaling producer test with "\
                     "tensor size of " << n_bytes <<
                     " bytes and "<< tensors_per_dataset <<
                     " tensors per dataset." << std::endl;

    run_aggregation_production(n_bytes, tensors_per_dataset);

    if(rank==0)
        std::cout << "Finished data aggregation production." << std::endl;

    MPI_Finalize();

    return 0;
}
