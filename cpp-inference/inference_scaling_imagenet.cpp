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
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::string context("Inference MPI Rank: " + std::to_string(rank));
  log_data(context, LLDebug, "Initialized rank");

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
  log_data(context, LLDebug, "client() time stored");

  std::string device = get_device();
  int num_devices = get_num_devices();
  bool use_multigpu = (0 == device.compare("GPU")) && num_devices > 1;
  bool should_set = get_set_flag();

  std::string model_key = "resnet_model";
  bool poll_model_code = client.poll_model(model_key, 100, 100);
  if (!poll_model_code) {
    log_error(context, LLInfo, "SR Error finding model");
  }

  std::string script_key = "resnet_script";
  bool poll_script_code = client.poll_key(script_key, 100, 100);
  if (!poll_script_code) {
    log_error(context, LLInfo, "SR Error finding script");
  }

  // setting up string to debug set vars
  std::string program_vars = "Running rank with vars should_set: ";
  program_vars += std::to_string(should_set) + " - num_device: ";
  program_vars += std::to_string(num_devices);
  program_vars += " - use_multigpu: " + std::to_string(use_multigpu) + " - is_coloated: ";
  program_vars += std::to_string(is_colocated) + " - cluster: " + std::to_string(cluster);
  log_data(context, LLDebug, program_vars);

  int iterations = get_iterations();
  log_data(context, LLDebug, "Running with iterations: " + std::to_string(iterations));
  MPI_Barrier(MPI_COMM_WORLD);
  log_data(context, LLDebug, "All scripts and models have been set");

  //Allocate a continugous memory to make bcast easier
  float* p = (float*)malloc(224*224*3*sizeof(float));
  if(rank == 0)
    load_mnist_image_to_array(p);
  MPI_Bcast(&(p[0]), 224*224*3, MPI_FLOAT, 0, MPI_COMM_WORLD);
  float*** array = allocate_3D_array<float>(224, 224, 3);
  int c = 0;

  // Fill array with random numbers. This array represents an 224x224 image
  // with 3 color channels
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
    log_data(context, LLInfo, "All ranks have Resnet image");
    std::cout<<"All ranks have Resnet image"<<std::endl;

  // Warmup the database with an inference
  std::string in_key = "resnet_input_rank_" + std::to_string(rank) + "_" + std::to_string(-1);
  std::string script_out_key = "resnet_processed_input_rank_" + std::to_string(rank) + "_" + std::to_string(-1);
  std::string out_key = "resnet_output_rank_" + std::to_string(rank) + "_" + std::to_string(-1);
  client.put_tensor(in_key, array, {224, 224, 3},
                    SRTensorTypeFloat,
                    SRMemLayoutNested);
  log_data(context, LLDebug, "put_tensor completed");
  if (use_multigpu) {
    client.run_script_multigpu(script_key, "pre_process_3ch", {in_key}, {script_out_key}, rank, 0, num_devices);
    std::string use_multi_run_script_vars = "Use_multigpu=True - script_key:" + script_key;
    use_multi_run_script_vars += " in_key:" + in_key + " script_out_key:";
    use_multi_run_script_vars += script_out_key + " rank:" + std::to_string(rank);
    use_multi_run_script_vars += " num_devices:" + std::to_string(num_devices);
    log_data(context, LLDebug, use_multi_run_script_vars);
  }
  else {
    client.run_script(script_key, "pre_process_3ch", {in_key}, {script_out_key});
    std::string run_script_vars = "Use_multigpu=False - script_key:" + script_key;
    run_script_vars += " in_key:" + in_key + " script_out_key:" + script_out_key;
    log_data(context, LLDebug, run_script_vars);
  }
  if (use_multigpu) {
    client.run_model_multigpu(model_key, {script_out_key}, {out_key}, rank, 0, num_devices);
    std::string use_multi_run_model_vars = "Use_multigpu=True - model_key:" + model_key;
    use_multi_run_model_vars += " out_key:" + out_key + " script_out_key:" + script_out_key;
    use_multi_run_model_vars += " rank:" + std::to_string(rank) = " num_devices:";
    use_multi_run_model_vars += std::to_string(num_devices);
    log_data(context, LLDebug, use_multi_run_model_vars);
  }
  else {
    client.run_model(model_key, {script_out_key}, {out_key});
    std::string run_model_vars = "Use_multigpu=False - model_key:" + model_key + " out_key:";
    run_model_vars += out_key + " script_out_key:" + script_out_key;
    log_data(context, LLDebug, run_model_vars);
  }
  client.unpack_tensor(out_key, result, {1,1000},
      SRTensorTypeFloat,
      SRMemLayoutNested);

  // Begin the actual iteration loop
  log_data(context, LLDebug, "Iteration loop starting...");
  MPI_Barrier(MPI_COMM_WORLD);
  double loop_start = MPI_Wtime();
  for (int i = 0; i < iterations + 1; i++) {
    log_data(context, LLDebug, "Running iteration: " + std::to_string(i));
    std::string in_key = "resnet_input_rank_" + std::to_string(rank) + "_" + std::to_string(i);
    std::string script_out_key = "resnet_processed_input_rank_" + std::to_string(rank) + "_" + std::to_string(i);
    std::string out_key = "resnet_output_rank_" + std::to_string(rank) + "_" + std::to_string(i);
    double put_tensor_start = MPI_Wtime();
    client.put_tensor(in_key, array, {224, 224, 3},
                      SRTensorTypeFloat,
                      SRMemLayoutNested);
    double put_tensor_end = MPI_Wtime();
    log_data(context, LLDebug, "put_tensor completed");
    delta_t = put_tensor_end - put_tensor_start;
    put_tensor_times.push_back(delta_t);

    double run_script_start = MPI_Wtime();
    if (use_multigpu) {
      client.run_script_multigpu(script_key, "pre_process_3ch", {in_key}, {script_out_key}, rank, 0, num_devices);
      std::string use_multi_run_script_vars_2 = "Use_multigpu=True - script_key: ";
      use_multi_run_script_vars_2 += script_key + " in_key: " + "script_out_key: ";
      use_multi_run_script_vars_2 =  script_out_key + " rank: " + std::to_string(rank);
      use_multi_run_script_vars_2 = " num_devices: " + std::to_string(num_devices);
      log_data(context, LLDebug, use_multi_run_script_vars_2);
    }
    else {
      client.run_script(script_key, "pre_process_3ch", {in_key}, {script_out_key});
      std::string run_script_vars_2 = "Use_multigpu=False - script_key: " + script_key;
      run_script_vars_2 += " in_key: " + in_key + "script_out_key: " + script_out_key;
      log_data(context, LLDebug, run_script_vars_2);
    }
    double run_script_end = MPI_Wtime();
    log_data(context, LLDebug, "run_script completed");
    delta_t = run_script_end - run_script_start;
    run_script_times.push_back(delta_t);

    double run_model_start = MPI_Wtime();
    if (use_multigpu) {
      client.run_model_multigpu(model_key, {script_out_key}, {out_key}, rank, 0, num_devices);
      std::string use_multi_run_model_vars_2 = "Use_multigpu=True - model_key: ";
      use_multi_run_model_vars_2 += model_key + " out_key: " + out_key;
      use_multi_run_model_vars_2 += "script_out_key: " + script_out_key + " rank: ";
      use_multi_run_model_vars_2 += std::to_string(rank) + " num_devices: ";
      use_multi_run_model_vars_2 += std::to_string(num_devices);
      log_data(context, LLDebug, use_multi_run_model_vars_2);
    }
    else {
      client.run_model(model_key, {script_out_key}, {out_key});
      std::string run_model_vars_2 = "Use_multigpu=False - model_key: " + model_key;
      run_model_vars_2 += " in_key: " + out_key + "script_out_key: " + script_out_key;
      log_data(context, LLDebug, run_model_vars_2);
    }
    double run_model_end = MPI_Wtime();
    log_data(context, LLDebug, "run_model completed");
    delta_t = run_model_end - run_model_start;
    run_model_times.push_back(delta_t);

    double unpack_tensor_start = MPI_Wtime();
    client.unpack_tensor(out_key, result, {1,1000},
			 SRTensorTypeFloat,
			 SRMemLayoutNested);
    double unpack_tensor_end = MPI_Wtime();
    log_data(context, LLDebug, "unpack_tensor completed");
    delta_t = unpack_tensor_end - unpack_tensor_start;
    unpack_tensor_times.push_back(delta_t);

  }
  double loop_end = MPI_Wtime();
  log_data(context, LLDebug, "Loop completed");
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
  log_data(context, LLDebug, "Times written to file");
  free_3D_array(array, 3, 224);
  free_2D_array(result, 1);
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  double main_start = MPI_Wtime();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::string context("Inference Tests Rank: " + std::to_string(rank));
  log_data(context, LLDebug, "Rank Initialized");

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
  log_data(context, LLDebug, "main() time stored");

  MPI_Finalize();
  log_data(context, LLDebug, "All ranks finalized");

  return 0;
}