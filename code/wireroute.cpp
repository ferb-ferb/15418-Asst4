/**
 * Parallel VLSI Wire Routing via MPI
 * Name 1(andrew_id 1), Name 2(andrew_id 2)
 */

#include "wireroute.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <mpi.h>
#include <unistd.h>

void print_stats(const std::vector<std::vector<int>> &occupancy) {
  int max_occupancy = 0;
  long long total_cost = 0;

  for (const auto &row : occupancy) {
    for (const int count : row) {
      max_occupancy = std::max(max_occupancy, count);
      total_cost += count * count;
    }
  }

  std::cout << "Max occupancy: " << max_occupancy << '\n';
  std::cout << "Total cost: " << total_cost << '\n';
}

/* This function write the output into 2 files
(1) It write occupancy grids into a file
(2) It convert wires from Wire to validate_wire_t by to_validate_format
(2) It write wires into another file
*/
void write_output(
    const std::vector<Wire> &wires, const int num_wires,
    const std::vector<std::vector<int>> &occupancy, const int dim_x,
    const int dim_y,
    std::string wires_output_file_path = "outputs/wire_output.txt",
    std::string occupancy_output_file_path = "outputs/occ_output.txt") {

  std::ofstream out_occupancy(occupancy_output_file_path, std::fstream::out);
  if (!out_occupancy) {
    std::cerr << "Unable to open file: " << occupancy_output_file_path << '\n';
    exit(EXIT_FAILURE);
  }
  out_occupancy << dim_x << ' ' << dim_y << '\n';

  for (const auto &row : occupancy) {
    for (size_t i = 0; i < row.size(); ++i)
      out_occupancy << row[i] << (i == row.size() - 1 ? "" : " ");
    out_occupancy << '\n';
  }
  out_occupancy.close();

  std::ofstream out_wires(wires_output_file_path, std::fstream::out);
  if (!out_wires) {
    std::cerr << "Unable to open file: " << wires_output_file_path << '\n';
    exit(EXIT_FAILURE);
  }

  out_wires << dim_x << ' ' << dim_y << '\n';
  out_wires << num_wires << '\n';

  for (const auto &wire : wires) {
    // NOTICE: we convert to keypoint representation here, using
    // to_validate_format which need to be defined in the bottom of this file
    validate_wire_t keypoints = wire.to_validate_format();
    for (int i = 0; i < keypoints.num_pts; ++i) {
      out_wires << keypoints.p[i].x << ' ' << keypoints.p[i].y;
      if (i < keypoints.num_pts - 1)
        out_wires << ' ';
    }
    out_wires << '\n';
  }

  out_wires.close();
}

int calc_cost(const Wire& wire, std::vector<std::vector<int>>& map, int mode) {
    int cost = 0;

    auto apply = [&](int x, int y) {
        if(mode == 0) {
            cost += (map[y][x]+1) * (map[y][x]+1);
        } else if(mode == 1) {
            map[y][x]++;
            cost += map[y][x] * map[y][x];
        } else if(mode == 2) {
            cost += map[y][x] * map[y][x];
            map[y][x]--;
        }
    };

    // Start -> Mid
    if(wire.move_x_start) {
        for(int x = std::min(wire.start_x, wire.mid_x); x <= std::max(wire.start_x, wire.mid_x); x++)
            apply(x, wire.start_y);
        for(int y = std::min(wire.start_y, wire.mid_y); y <= std::max(wire.start_y, wire.mid_y); y++)
            apply(wire.mid_x, y);
        switch(mode){
        case(0):
          break;
        case(1):
          map[wire.start_y][wire.mid_x]--;
          break;
        case(2):
          map[wire.start_y][wire.mid_x]++;
          break;
        }
    } else {
        for(int y = std::min(wire.start_y, wire.mid_y); y <= std::max(wire.start_y, wire.mid_y); y++)
            apply(wire.start_x, y);
        for(int x = std::min(wire.start_x, wire.mid_x); x <= std::max(wire.start_x, wire.mid_x); x++)
            apply(x, wire.mid_y);
        switch(mode){
        case(0):
          break;
        case(1):
          map[wire.mid_y][wire.start_x]--;
          break;
        case(2):
          map[wire.mid_y][wire.start_x]++;
          break;
        }
    }

    // Mid -> End
    if(wire.move_x_end) {
        for(int x = std::min(wire.mid_x, wire.end_x); x <= std::max(wire.mid_x, wire.end_x); x++)
            apply(x, wire.mid_y);
        for(int y = std::min(wire.mid_y, wire.end_y); y <= std::max(wire.mid_y, wire.end_y); y++)
            apply(wire.end_x, y);
        switch(mode){
        case(0):
          break;
        case(1):
          map[wire.mid_y][wire.mid_x]--;
          map[wire.mid_y][wire.end_x]--;
          break;
        case(2):
          map[wire.mid_y][wire.mid_x]++;
          map[wire.mid_y][wire.end_x]++;
          break;
        }
    } else {
        for(int y = std::min(wire.mid_y, wire.end_y); y <= std::max(wire.mid_y, wire.end_y); y++)
            apply(wire.mid_x, y);
        for(int x = std::min(wire.mid_x, wire.end_x); x <= std::max(wire.mid_x, wire.end_x); x++)
            apply(x, wire.end_y);
        switch(mode){
        case(0):
          break;
        case(1):
          map[wire.mid_y][wire.mid_x]--;
          map[wire.end_y][wire.mid_x]--;
          break;
        case(2):
          map[wire.mid_y][wire.mid_x]++;
          map[wire.end_y][wire.mid_x]++;
          break;
        }
    }
    return cost;
}
Wire find_best_route(const Wire& wire, std::vector<std::vector<int>>& occ,
                    std::mt19937& rng, double SA_prob) {
    if (wire.start_x == wire.end_x || wire.start_y == wire.end_y)
        return wire;

    int dx_min = std::min(wire.start_x, wire.end_x);
    int dx_max = std::max(wire.start_x, wire.end_x);
    int dy_min = std::min(wire.start_y, wire.end_y);
    int dy_max = std::max(wire.start_y, wire.end_y);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    if (prob_dist(rng) < SA_prob) {
    std::uniform_int_distribution<int> x_dist(dx_min, dx_max);
    std::uniform_int_distribution<int> y_dist(dy_min, dy_max);
    std::uniform_int_distribution<int> orient_dist(0, 1);

    Wire candidate = wire;
    candidate.mid_x = x_dist(rng);
    candidate.mid_y = y_dist(rng);
    candidate.move_x_start = orient_dist(rng);
    candidate.move_x_end   = candidate.move_x_start;
    
    return candidate;
    }   
    int best_cost = INT_MAX;
    Wire best_wire = wire;

    // current route
    { int cost = calc_cost(wire, occ, 0); if (cost < best_cost) { best_cost = cost; best_wire = wire; } }

    // x-first
    for (int x = dx_min+1; x <= dx_max; x++) {
        Wire c = wire; c.mid_x = x; c.mid_y = wire.end_y;
        c.move_x_start = true; c.move_x_end = false;
        int cost = calc_cost(c, occ, 0);
        if (cost < best_cost) { best_cost = cost; best_wire = c; }
    }
    // y-first
    for (int y = dy_min+1; y <= dy_max; y++) {
        Wire c = wire; c.mid_x = wire.end_x; c.mid_y = y;
        c.move_x_start = false; c.move_x_end = true;
        int cost = calc_cost(c, occ, 0);
        if (cost < best_cost) { best_cost = cost; best_wire = c; }
    }
    // 3-bend
    for (int x = dx_min+1; x < dx_max; x++) {
        for (int y = dy_min+1; y < dy_max; y++) {
            { Wire c = wire; c.mid_x=x; c.mid_y=y; c.move_x_start=true;  c.move_x_end=true;
              int cost = calc_cost(c,occ,0); if(cost<best_cost){best_cost=cost;best_wire=c;} }
            { Wire c = wire; c.mid_x=x; c.mid_y=y; c.move_x_start=false; c.move_x_end=false;
              int cost = calc_cost(c,occ,0); if(cost<best_cost){best_cost=cost;best_wire=c;} }
        }
    }
    return best_wire;
}


int main(int argc, char *argv[]) {
  const auto init_start = std::chrono::steady_clock::now();
  int pid;
  int nproc;

  // Initialize MPI
  MPI_Init(&argc, &argv);
  // Get process rank
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  // Get total number of processes  
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  std::string input_filename;
  double SA_prob = 0.1;
  int SA_iters = 5;
  int batch_size = 1;

  // Read command line arguments
  int opt;
  while ((opt = getopt(argc, argv, "f:p:i:m:b:")) != -1) {
    switch (opt) {
      case 'f':
        input_filename = optarg;
        break;
      case 'p':
        SA_prob = atof(optarg);
        break;
      case 'i':
        SA_iters = atoi(optarg);
        break;
      case 'b':
        batch_size = atoi(optarg);
        break;
      default:
        if (pid == 0) {
          std::cerr << "Usage: " << argv[0] << " -f input_filename [-p SA_prob] [-i SA_iters] -b batch_size\n";
        }

        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
  }

  // Check if required options are provided
  if (empty(input_filename) || SA_iters <= 0 || batch_size <= 0) {
    if (pid == 0) {
      std::cerr << "Usage: " << argv[0] << " -f input_filename [-p SA_prob] [-i SA_iters] -b batch_size\n";
    }

    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  if (pid == 0) {
    std::cout << "Number of processes: " << nproc << '\n';
    std::cout << "Simulated annealing probability parameter: " << SA_prob << '\n';
    std::cout << "Simulated annealing iterations: " << SA_iters << '\n';
    std::cout << "Input file: " << input_filename << '\n';
    std::cout << "Batch size: " << batch_size << '\n';
  }

  int dim_x, dim_y, num_wires;
  std::vector<Wire> wires;
  std::vector<std::vector<int>> occupancy;

  if (pid == 0) {
      std::ifstream fin(input_filename);

      if (!fin) {
        std::cerr << "Unable to open file: " << input_filename << ".\n";
        exit(EXIT_FAILURE);
      }

      /* Read the grid dimension and wire information from file */
      fin >> dim_x >> dim_y >> num_wires;

      wires.resize(num_wires);
      for (auto &wire : wires) {
        fin >> wire.start_x >> wire.start_y >> wire.end_x >> wire.end_y;
        wire.move_x_start = true;
        wire.move_x_end = false;
        wire.mid_x = wire.end_x;
        wire.mid_y = wire.start_y;
      }
  }

  /* Initialize any additional data structures needed in the algorithm */

  if (pid == 0) {
    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';
  }

  const auto compute_start = std::chrono::steady_clock::now();

  /** 
   * (TODO)
   * Implement the wire routing algorithm here
   * Feel free to structure the algorithm into different functions
   * Use MPI to parallelize the algorithm. 
   */

  if (pid == 0) {
    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << std::fixed << std::setprecision(10) << compute_time << '\n';
  }

  if (pid == 0) {
    /* Write wires and occupancy matrix to files */
    print_stats(occupancy);
    write_output(wires, num_wires, occupancy, dim_x, dim_y);
  }

  // Cleanup
  MPI_Finalize();
}

/* TODO (student): implement to_validate_format to convert Wire to
  validate_wire_t keypoint representation in order to run checker and
  write output
*/
validate_wire_t Wire::to_validate_format(void) const {
  validate_wire_t w;
  w.num_pts = 1;
  w.p[0].x = this -> start_x;
  w.p[0].y = this -> start_y;
  if(this->move_x_start){
    if(this->start_x != this->mid_x){
      w.p[w.num_pts].x = this->mid_x;
      w.p[w.num_pts].y = this->start_y;
      w.num_pts++;
    }
    if(this->start_y != this->mid_y){
      w.p[w.num_pts].x = this -> mid_x;
      w.p[w.num_pts].y = this -> mid_y; 
      w.num_pts++;
    }
  }
  else{
    if(this->start_y != this->mid_y){
      w.p[w.num_pts].x = this -> start_x;
      w.p[w.num_pts].y = this -> mid_y; 
      w.num_pts++;
    }
    if(this->start_x != this->mid_x){
      w.p[w.num_pts].x = this->mid_x;
      w.p[w.num_pts].y = this->mid_y;
      w.num_pts++;
    }
  }
  if(this->move_x_end){
    if(this->mid_x != this->end_x){
      w.p[w.num_pts].x = this->end_x;
      w.p[w.num_pts].y = this->mid_y;
      w.num_pts++;
    }
    if(this->mid_y != this->end_y){
      w.p[w.num_pts].x = this -> end_x;
      w.p[w.num_pts].y = this -> end_y; 
      w.num_pts++;
    }
  }
  else{
    if(this->mid_y != this->end_y){
      w.p[w.num_pts].x = this -> mid_x;
      w.p[w.num_pts].y = this -> end_y; 
      w.num_pts++;
    }
    if(this->mid_x != this->end_x){
      w.p[w.num_pts].x = this->end_x;
      w.p[w.num_pts].y = this->end_y;
      w.num_pts++;
    }
  }
  return w;
}
