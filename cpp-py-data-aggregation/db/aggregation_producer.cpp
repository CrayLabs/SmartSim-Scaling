#include "client.h"
#include <mpi.h>


int get_iterations() {
  char* iterations = std::getenv("SS_ITERATIONS");
  int iters = iterations ? std::stoi(iterations) : 20;
  return iters;
}

void run_aggregation_production(size_t n_bytes,
                                size_t tensors_per_dataset)
{

    //Initializing rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::string context("Data Aggregation PY MPI Producer Rank: " + std::to_string(rank));
    log_data(context, LLDebug, "Initialized rank");

    //Indicate Client creation
    if (rank == 0)
        log_data(context, LLInfo, "Connecting clients");
        std::cout << "Connecting clients" << std::endl;

    // Connect a client for each MPI rank
    SmartRedis::Client client(true, context);

    // Block for all clients to be connected
    MPI_Barrier(MPI_COMM_WORLD);

    // Create tensors for the dataset
    size_t n_values = n_bytes / sizeof(float);
    std::vector<float> array(n_values, 0);
    std::vector<float> result(n_values, 0);
    for(size_t i = 0; i < n_values; i++)
        array[i] = i;

    // Get the number of iterations to perform
    int iterations = get_iterations();
    log_data(context, LLDebug, "Running with iterations: " + std::to_string(iterations));

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
        log_data(context, LLDebug, "Running iteration: " + std::to_string(i));

        // Set the list name (not MPI rank dependent)
        std::string list_name = "iteration_" + std::to_string(i);

        // This poll invocation syncs the producer to wait for the
        // consumer to finish the previous iteration aggregation
        if (rank == 0 && i != 0) {
            std::string last_list_name = "iteration_" + std::to_string(i - 1);
            while (client.get_list_length(last_list_name) != 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
        }

        // Block until the consumer has deleted the pervious iteration list
        MPI_Barrier(MPI_COMM_WORLD);

        // Put and Append to the aggregation list
        client.put_dataset(dataset);
        client.append_to_list(list_name, dataset);
    }
}

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    //initializing rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::string context("Data Aggregation PY Tests Rank " + std::to_string(rank));
    log_data(context, LLDebug, "Rank initialized");

    // Get command line arguments
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

    if(rank==0) {
        std::string if_rank_0 = "Running aggregate scaling producer test with ";
        if_rank_0 += "tensor size of " + std::to_string(n_bytes) + " bytes and ";
        if_rank_0 += std::to_string(tensors_per_dataset) + " tensors per dataset.";
        log_data(context, LLInfo, if_rank_0);
        std::cout << "Running aggregate scaling producer test with "\
                     "tensor size of " << n_bytes <<
                     " bytes and "<< tensors_per_dataset <<
                     " tensors per dataset." << std::endl;
    }
    // Run the dataset and aggregation list production
    run_aggregation_production(n_bytes, tensors_per_dataset);

    if(rank==0)
        log_data(context, LLInfo, "Finished data aggregation production.");
        std::cout << "Finished data aggregation production." << std::endl;

    MPI_Finalize();

    return 0;
}
