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
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);//this is the specific image - the rank

    if(!rank)
        std::cout<<"Connecting clients"<<std::endl<<std::flush;

    double constructor_start = MPI_Wtime();//gets the current time on the machine
    bool cluster = get_cluster_flag();
    SmartRedis::Client client(cluster);
    double constructor_end = MPI_Wtime();
    double delta_t = constructor_end - constructor_start;
    timing_file << rank << "," << "client()" << ","
                << delta_t << "\n";

    MPI_Barrier(MPI_COMM_WORLD);//makes sure all the clients are running and setup, all the copies of the program need to hit this line before continuing
    //who has called this function
    size_t n_values = n_bytes / sizeof(float);//size t is used in C++ to store size of types
    std::vector<float> array(n_values, 0);//std determines what name space we r looking for the objects in
    std::vector<float> result(n_values, 0);
    for(size_t i=0; i<n_values; i++)
        array[i] = i;


    // allocate arrays to hold timings
    std::vector<double> put_tensor_times;
    std::vector<double> unpack_tensor_times;//tensor is end dim array

    int iterations = get_iterations();

    MPI_Barrier(MPI_COMM_WORLD);

    double loop_start = MPI_Wtime();

    // Keys are overwritten in order to help
    // ensure that the database does not run out of memory
    // for large messages.
    for (int i=0; i<iterations; i++) {

        std::string key = "throughput_rank_" +
                          std::to_string(rank);

        double put_tensor_start = MPI_Wtime();
        //put tensor says Ive got an array and I want to send to db, so puts into db
        client.put_tensor(key,
                          array.data(),
                          {1, n_values},//dimension of the tensor
                          SRTensorTypeFloat, //different type of tensors
                          SRMemLayoutContiguous);
        double put_tensor_end = MPI_Wtime();
        delta_t = put_tensor_end - put_tensor_start;
        put_tensor_times.push_back(delta_t);//put this value in the first space of the vector

        double unpack_tensor_start = MPI_Wtime();
        client.unpack_tensor(key,
                             result.data(),
                             {n_values},
                             SRTensorTypeFloat,
                             SRMemLayoutContiguous);
        double unpack_tensor_end = MPI_Wtime();
        delta_t = unpack_tensor_end - unpack_tensor_start;
        unpack_tensor_times.push_back(delta_t);//throughput things
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

    timing_file << std::flush;//if there is anything in the buffer, write it all to disk
    //add MPI Barrier
    return;
}

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);//sets up the parallel communications, setsup the actual communicator that all the images will use to talk to eachohter

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
    std::ofstream timing_file;//open file stream
    timing_file.open("rank_" + std::to_string(rank) + "_timing.csv");

    run_throughput(timing_file, n_bytes);

    if(rank==0)
        std::cout<<"Finished throughput test."<<std::endl;

    double main_end = MPI_Wtime();
    double delta_t = main_end - main_start;
    timing_file << rank << "," << "main()" << ","
                << delta_t << std::endl << std::flush;

    MPI_Finalize(); //tears down the communcator, tears down what init makes

    return 0;
}
