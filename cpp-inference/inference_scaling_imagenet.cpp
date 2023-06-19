#include "client.h"
#include "client_test_utils.h"
#include <algorithm>
#include <mpi.h>



// some helpers for handling model settings from the
// driver script in colocated and non-colocated cases
int get_iterations() {
  char* iterations = std::getenv("SS_ITERATIONS");
  int iters = iterations ? std::stoi(iterations) : 100;
  return iters;
}

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

bool get_colo() {
  char* is_colocated = std::getenv("SS_COLOCATED");
  return is_colocated ? std::stoi(is_colocated) : false;
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
  std::string context("Run Inference");
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  log_data(context, LLDebug, "rank: " + std::to_string(rank) + " initiated");

  if (!rank)
    log_data(context, LLInfo, "Connecting clients");
    std::cout<<"Connecting clients"<<std::endl<<std::flush;

  double constructor_start = MPI_Wtime();
  bool cluster = get_cluster_flag();
  bool is_colocated = get_colo();
  SmartRedis::Client client(cluster, context);
  double constructor_end = MPI_Wtime();
  double delta_t = constructor_end - constructor_start;
  timing_file << rank << "," << "client()" << ","
              << delta_t << std::endl << std::flush;
  //print rank # storing client() for debugging
  std::string client_text = "client() time stored for rank: ";
  client_text += std::to_string(rank);
  log_data(context, LLDebug, client_text);

  std::string device = get_device();
  int num_devices = get_num_devices();
  bool use_multigpu = (0 == device.compare("GPU")) && num_devices > 1;
  bool should_set = get_set_flag();
  std::string model_key = "resnet_model";
  std::string script_key = "resnet_script";
  log_data(context, LLDebug, "Running rank: " + std::to_string(rank) + "with vars should_set: " + std::to_string(should_set) + " - num_device: " + std::to_string(num_devices) + " - use_multigpu: " + std::to_string(use_multigpu) + " - is_coloated: " + std::to_string(is_colocated) + " - cluster: " + std::to_string(cluster));
  if (should_set) {
    int batch_size = get_batch_size();
    int n_clients = get_client_count();
    log_date(context, LLDebug, "Running rank: " + std::to_string(rank) + "with batch_size: " + std::to_string(batch_size) + " and n_clients: " + std::to_string(n_clients));
    if (!is_colocated && rank == 0) {
      log_data(context, LLDebug, "Rank: " + std::to_string(rank) + " entered non-colocated if block");

      std::cout<<"Setting Resnet Model from scaling app" << std::endl;
      log_data(context, LLInfo, "Setting Resnet Model from scaling app");

      std::cout<<"Setting with batch_size: " << std::to_string(batch_size) << std::endl;
      std::string batch_text = "Setting with batch_size: ";
      batch_text += std::to_string(batch_size);
      log_data(context, LLInfo, batch_text);

      std::cout<<"Setting on device: " << device << std::endl;
      std::string device_text = "Setting on device: ";
      device_text += device;
      log_data(context, LLInfo, device_text);

      std::cout<<"Setting on " << std::to_string(num_devices) << " devices" <<std::endl << std::flush;
      std::string set_text = "Setting on ";
      set_text += std::to_string(num_devices);
      set_text += " devices";
      log_data(context, LLInfo, set_text);

      std::string model_filename = "./resnet50." + device + ".pt";

      if (use_multigpu) {
        log_data(context, LLDebug, "model_key: " + model_key + " model_filename: " + model_filename + " num_devices: " + std::to_string(num_devices) + " batch_size: " + std::to_string(batch_size));
        client.set_model_from_file_multigpu(model_key, model_filename, "TORCH", 0, num_devices, batch_size);
        client.set_script_from_file_multigpu(script_key, "./data_processing_script.txt", 0, num_devices);
      }
      //HERE
      else {
          log_data(context, LLDebug, "model_key: " + model_key + " model_filename: " + model_filename + " device: " + std::to_string(device) + " batch_size: " + std::to_string(batch_size));
          client.set_model_from_file(model_key, model_filename, "TORCH", device, batch_size);
          client.set_script_from_file(script_key, device, "./data_processing_script.txt");
      }
    }
    if(is_colocated && rank % n_clients == 0) {
      log_data(context, LLDebug, "Rank: " + std::to_string(rank) + " entered colocated if block");

      std::cout<<"Setting Resnet Model from scaling app" << std::endl;
      log_data(context, LLInfo, "Setting Resnet Model from scaling app");

      std::cout<<"Setting with batch_size: " << std::to_string(batch_size) << std::endl;
      std::string batch_text = "Setting with batch_size: ";
      batch_text += std::to_string(batch_size);
      log_data(context, LLInfo, batch_text);

      std::cout<<"Setting on device: " << device << std::endl;
      std::string device_text = "Setting on device: ";
      device_text += device;
      log_data(context, LLInfo, device_text);

      std::cout<<"Setting on " << std::to_string(num_devices) << " devices" <<std::endl << std::flush;
      std::string set_text = "Setting on ";
      set_text += std::to_string(num_devices);
      set_text += " devices";
      log_data(context, LLInfo, set_text);

      std::string model_filename = "./resnet50." + device + ".pt";

      if (use_multigpu) {
        client.set_model_from_file_multigpu(model_key, model_filename, "TORCH", 0, num_devices, batch_size);
        client.set_script_from_file_multigpu(script_key, "./data_processing_script.txt", 0, num_devices);
      }
      else {
          client.set_model_from_file(model_key, model_filename, "TORCH", device, batch_size);
          client.set_script_from_file(script_key, device, "./data_processing_script.txt");
      }
    }
  }
  int iterations = get_iterations();
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
        array[i][j][k] = (rand() % 100)*0.01;
      }

  //float**** array = allocate_4D_array<float>(1,1,28,28);
  float** result = allocate_2D_array<float>(1, 1000);

  std::vector<double> put_tensor_times;
  std::vector<double> run_script_times;
  std::vector<double> run_model_times;
  std::vector<double> unpack_tensor_times;

  if (!rank)
    std::cout<<"All ranks have Resnet image"<<std::endl;

  // Warmup the database with an inference
  std::string in_key = "resnet_input_rank_" + std::to_string(rank) + "_" + std::to_string(-1);
  std::string script_out_key = "resnet_processed_input_rank_" + std::to_string(rank) + "_" + std::to_string(-1);
  std::string out_key = "resnet_output_rank_" + std::to_string(rank) + "_" + std::to_string(-1);
  client.put_tensor(in_key, array, {224, 224, 3},
                    SRTensorTypeFloat,
                    SRMemLayoutNested);
  double put_tensor_end = MPI_Wtime();

  if (use_multigpu)
    client.run_script_multigpu(script_key, "pre_process_3ch", {in_key}, {script_out_key}, rank, 0, num_devices);
  else
    client.run_script(script_key, "pre_process_3ch", {in_key}, {script_out_key});
  if (use_multigpu)
    client.run_model_multigpu(model_key, {script_out_key}, {out_key}, rank, 0, num_devices);
  else
    client.run_model(model_key, {script_out_key}, {out_key});
  client.unpack_tensor(out_key, result, {1,1000},
      SRTensorTypeFloat,
      SRMemLayoutNested);

  // Begin the actual iteration loop
  double loop_start = MPI_Wtime();
  for (int i = 0; i < iterations + 1; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
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
  for (int i = 1; i < iterations; i++) { // Skip first run as it's warmup
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
  std::string context("Inference Scaling Tests");
  MPI_Init(&argc, &argv);

  log_data(context, LLDebug, "Starting Inference tests");
  double main_start = MPI_Wtime();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  log_data(context, LLDebug, "rank: " + std::to_string(rank) + " initiated");

  //Open Timing file
  std::ofstream timing_file;
  timing_file.open("rank_"+std::to_string(rank)+"_timing.csv");

  // Run timing tests
  run_mnist("resnet_model", "resnet_script", timing_file);
  if(rank==0)
    log_data(context, LLInfo, "Finished Resnet test.");
    std::cout<<"Finished Resnet test."<<std::endl;

  double main_end = MPI_Wtime();
  double delta_t = main_end - main_start;
  timing_file << rank << "," << "main()" << ","
              << delta_t << std::endl << std::flush;
  //print rank # storing client() for debugging
  std::string main_text = "main() time stored for rank: ";
  main_text += std::to_string(rank);
  log_data(context, LLDebug, main_text);

  MPI_Finalize();
  log_data(context, LLDebug, "All ranks finalized");

  return 0;
}