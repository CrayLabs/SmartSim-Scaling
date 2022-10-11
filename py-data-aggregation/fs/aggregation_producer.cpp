#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <unordered_map>
#include <thread>
#include <stdexcept>
#include <mpi.h>


namespace fs = std::filesystem;

int get_iterations() {
  char* iterations = std::getenv("SS_ITERATIONS");
  int iters = iterations ? std::stoi(iterations) : 20;
  return iters;
}

fs::path get_write_to_dir() {
    char* write_to_dir = std::getenv("WRITE_TO_DIR");
    if (write_to_dir == nullptr)
        throw std::runtime_error("Do not know where to write out going data");
    return write_to_dir;
}

void run_aggregation_production(size_t n_bytes,
                                size_t tensors_per_dataset)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Block for all clients to be connected
    MPI_Barrier(MPI_COMM_WORLD);

    // Create tensors for the dataset
    size_t n_values = n_bytes / sizeof(float);
    std::vector<float> array(n_values, 0);
    std::vector<float> result(n_values, 0);
    for(size_t i = 0; i < n_values; i++)
        array[i] = i;


    // Get the number of iterations to perform

    // Set the dataset name (MPI rank dependent)
    std::string name = "aggregation_rank_" + std::to_string(rank);

    // fds for writing to buffer file
    std::ofstream fout(get_write_to_dir() / (name + ".dat"), std::ios::out | std::ios::binary);
    fout.open();

    // Create the dataset and add specified number of tensors
    std::unordered_map< std::string, std::vector<float> > dataset = {};
    for (size_t j = 0; j < tensors_per_dataset; j++) {
        std::string tensor_name = "tensor_" + std::to_string(j);
        dataset.insert({tensor_name, array});
    }

    // Write the dataset to the file system
    size_t num = name.size();
    fout.write(reinterpret_cast<const char *>(&num), sizeof(size_t));
    fout.write(name.c_str(), num * sizeof(char));
    for (const auto& [name, tensor] : dataset) {
        num = name.size();
        fout.write(reinterpret_cast<const char *>(&num), sizeof(size_t));
        fout.write(name.c_str(), num * sizeof(char));
        
        num = tensor.size();
        fout.write(reinterpret_cast<const char *>(&num), sizeof(size_t));
        fout.write(tensor.data(), num * sizeof(float));
    }
    fout.close();
    

    // A new list (dir of datasets) is created for each iteration
    // to measure dataset aggregation throughput
    std::string list_name = "iteration_" + std::to_string(0);
    fs::create_directory(get_write_to_dir() / list_name);
    for (int i = 0; i < iterations; i++) {

        // Set the list name (not MPI rank dependent)
        list_name = "iteration_" + std::to_string(i);

        // This poll invocation syncs the producer to wait for the
        // consumer to finish the previous iteration aggregation
        if (rank == 0 && i != 0) {
            std::string last_list_name = "iteration_" + std::to_string(i - 1);
            while (fs::exists(get_write_to_dir() / last_list_name)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
            fs::create_directory(get_write_to_dir() / list_name);
        }

        // Block until the consumer has deleted the pervious iteration list
        MPI_Barrier(MPI_COMM_WORLD);

        // Append to the aggregation list (put in dir)
        fs::copy_file(
            get_write_to_dir() / (name + ".dat"), 
            get_write_to_dir() / list_name / (name + ".dat"));
    }

    // Clean up data set still on file system
    fs::remove(get_write_to_dir() / (name + ".dat"));
}

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

    if(rank==0)
        std::cout << "Running aggregate scaling producer test with "\
                     "tensor size of " << n_bytes <<
                     " bytes and "<< tensors_per_dataset <<
                     " tensors per dataset." << std::endl;

    // Run the dataset and aggregation list production
    run_aggregation_production(n_bytes, tensors_per_dataset);

    if(rank==0)
        std::cout << "Finished data aggregation production." << std::endl;

    MPI_Finalize();

    return 0;
}
