#include "client.h"
#include "client_test_utils.h"
#include <algorithm>
#include <mpi.h>



// some helpers for handling model settings from the
// driver script in colocated and non-colocated cases
int get_batch_size() {
  char* batch_setting = std::getenv("SS_BATCH_SIZE");
  int batch_size = batch_setting ? std::stoi(batch_setting) : 1;
  return batch_size;
}

std::string get_device() {
  char* device = std::getenv("SS_DEVICE");
  std::string device_str = device ? device : "GPU";
  std::transform(device_str.begin(), device_str.end(), device_str.begin(), ::toupper);
  return device_str;
}

int get_num_devices() {
  char* num_dev_setting = std::getenv("SS_NUM_DEVICES");
  int num_devices = num_dev_setting ? std::stoi(num_dev_setting) : 1;
  return num_devices;
}

bool get_set_flag() {
  char* set_flag = std::getenv("SS_SET_MODEL");
  bool should_set = set_flag ? std::stoi(set_flag) : false;
  return should_set;
}

bool get_cluster_flag() {
  char* cluster_flag = std::getenv("SS_CLUSTER");
  bool use_cluster = cluster_flag ? std::stoi(cluster_flag) : false;
  return use_cluster;
}

int get_client_count() {
  char* client_count_flag = std::getenv("SS_CLIENT_COUNT");
  int client_count = client_count_flag ? std::stoi(client_count_flag) : 18;
  return client_count;
}


void load_mnist_image_to_array(float*& img)
{
  std::string image_file = "./cat.raw";
  std::ifstream fin(image_file, std::ios::binary);
  std::ostringstream ostream;
  ostream << fin.rdbuf();
  fin.close();
  const std::string tmp = ostream.str();
  const char *image_buf = tmp.data();
  int image_buf_length = tmp.length();
  std::memcpy(img, image_buf,
              image_buf_length*sizeof(char));
}

void run_mnist(const std::string& model_name,
               const std::string& script_name,
               std::ofstream& timing_file)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (!rank)
    std::cout<<"Connecting clients"<<std::endl<<std::flush;

  double constructor_start = MPI_Wtime();
  bool cluster = get_cluster_flag();
  SmartRedis::Client client(cluster);
  double constructor_end = MPI_Wtime();
  double delta_t = constructor_end - constructor_start;
  timing_file << rank << "," << "client()" << ","
              << delta_t << std::endl << std::flush;
  std::string device = get_device();
  int num_devices = get_num_devices();
  bool use_multigpu = (0 == device.compare("GPU")) && num_devices > 1;
  bool should_set = get_set_flag();
  if (should_set) {
    int batch_size = get_batch_size();
    int n_clients = get_client_count();
    if (rank % n_clients == 0) {
      std::cout<<"Setting Resnet Model from scaling app" << std::endl << std::flush;
      std::cout<<"Setting with batch_size: " << std::to_string(batch_size) << std::endl << std::flush;
      std::cout<<"Setting on device: " << device << std::endl << std::flush;
      std::cout<<"Setting on " << std::to_string(num_devices) << " devices" <<std::endl << std::flush;
      std::string model_filename = "./resnet50." + device + ".pt";

      if (use_multigpu) {
        std::string model_key = "resnet_model_0";
        std::string script_key = "resnet_script_0";
        client.set_model_from_file_multigpu(model_key, model_filename, "TORCH", 0, num_devices, batch_size);
        client.set_script_from_file_multigpu(script_key, "./data_processing_script.txt", 0, num_devices);
      }
      else {
        for (int i=0; i<num_devices; i++ ) {
          std::string model_key = "resnet_model_" + std::to_string(i);
          std::string script_key = "resnet_script_" + std::to_string(i);
          client.set_model_from_file(model_key, model_filename, "TORCH", device, batch_size);
          client.set_script_from_file(script_key, device, "./data_processing_script.txt");
        
        }
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  //Allocate a continugous memory to make bcast easier
  float* p = (float*)malloc(224*224*3*sizeof(float));
  if(rank == 0)
    load_mnist_image_to_array(p);
  MPI_Bcast(&(p[0]), 224*224*3, MPI_FLOAT, 0, MPI_COMM_WORLD);
  float*** array = allocate_3D_array<float>(224, 224, 3);
  int c = 0;
  for(int i=0; i<224; i++)
    for(int j=0; j<224; j++)
      for(int k=0; k<3; k++) {
        array[i][j][k] = p[c];
        c++;
      }

  //float**** array = allocate_4D_array<float>(1,1,28,28);
  float** result = allocate_2D_array<float>(1, 1000);

  std::vector<double> put_tensor_times;
  std::vector<double> run_script_times;
  std::vector<double> run_model_times;
  std::vector<double> unpack_tensor_times;

  // create keys for models and scripts to split inferences accross mulitple
  // CPU threads. This also covers the case for one GPU as rank%1 == 0
  std::string model_key = "resnet_model_" + std::to_string(rank%num_devices);
  std::string script_key = "resnet_script_" + std::to_string(rank%num_devices);
  // On GPU, we have the multigpu support, we don't need multiple names
  if (use_multigpu) {
    model_key = "resnet_model_0";
    script_key = "resnet_script_0";
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (!rank)
    std::cout<<"All ranks have Resnet image"<<std::endl;

  double loop_start = MPI_Wtime();
  for (int i = 0; i < 101; i++) {
    std::string in_key = "resnet_input_rank_" + std::to_string(rank) + "_" + std::to_string(i);
    std::string script_out_key = "resnet_processed_input_rank_" + std::to_string(rank) + "_" + std::to_string(i);
    std::string out_key = "resnet_output_rank_" + std::to_string(rank) + "_" + std::to_string(i);

    double put_tensor_start = MPI_Wtime();
    client.put_tensor(in_key, array, {224, 224, 3},
                      SRTensorTypeFloat,
                      SRMemLayoutNested);
    double put_tensor_end = MPI_Wtime();
    delta_t = put_tensor_end - put_tensor_start;
    put_tensor_times.push_back(delta_t);

    double run_script_start = MPI_Wtime();
    if (use_multigpu)
      client.run_script_multigpu(script_key, "pre_process_3ch", {in_key}, {script_out_key}, rank, 0, num_devices);
    else
      client.run_script(script_key, "pre_process_3ch", {in_key}, {script_out_key});
    double run_script_end = MPI_Wtime();
    delta_t = run_script_end - run_script_start;
    run_script_times.push_back(delta_t);

    double run_model_start = MPI_Wtime();
    if (use_multigpu)
      client.run_model_multigpu(model_key, {script_out_key}, {out_key}, rank, 0, num_devices);
    else
      client.run_model(model_key, {script_out_key}, {out_key});
    double run_model_end = MPI_Wtime();
    delta_t = run_model_end - run_model_start;
    run_model_times.push_back(delta_t);

    double unpack_tensor_start = MPI_Wtime();
    client.unpack_tensor(out_key, result, {1,1000},
			 SRTensorTypeFloat,
			 SRMemLayoutNested);
    double unpack_tensor_end = MPI_Wtime();
    delta_t = unpack_tensor_end - unpack_tensor_start;
    unpack_tensor_times.push_back(delta_t);

  }
  double loop_end = MPI_Wtime();
  delta_t = loop_end - loop_start;

  // write times to file
  for (int i = 1; i < 101; i++) { // Skip first run as it's warmup
    timing_file << rank << "," << "put_tensor" << ","
                << put_tensor_times[i] << std::endl << std::flush;

    timing_file << rank << "," << "run_script" << ","
                << run_script_times[i] << std::endl << std::flush;

    timing_file << rank << "," << "run_model" << ","
                << run_model_times[i] << std::endl << std::flush;

    timing_file << rank << "," << "unpack_tensor" << ","
                << unpack_tensor_times[i] << std::endl << std::flush;

  }
  timing_file << rank << "," << "loop_time" << ","
              << delta_t << std::endl << std::flush;

  free_3D_array(array, 3, 224);
  free_2D_array(result, 1);
}

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  double main_start = MPI_Wtime();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //Open Timing file
  std::ofstream timing_file;
  timing_file.open("rank_"+std::to_string(rank)+"_timing.csv");

  // Run timing tests
  run_mnist("resnet_model", "resnet_script", timing_file);
  if(rank==0)
    std::cout<<"Finished Resnet test."<<std::endl;

  double main_end = MPI_Wtime();
  double delta_t = main_end - main_start;
  timing_file << rank << "," << "main()" << ","
              << delta_t << std::endl << std::flush;

  MPI_Finalize();

  return 0;
}
