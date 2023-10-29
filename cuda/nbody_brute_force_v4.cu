/*
** nbody_brute_force.c - nbody simulation using the brute-force algorithm (O(n*n))
** NOTE: Implementation Using FP Precision.
**/ 

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef DISPLAY
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif

extern "C" {
#include "ui.h"
#include "nbody.h"
#include "nbody_tools.h"
}

//MACRO TO DEBUG CUDA FUNCTIONS
/** Error checking,
 *  taken from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

FILE* f_out=NULL;

int nparticles=10;      /* number of particles */
float T_FINAL=1.0;     /* simulation end time */
particle_t*particles;

float sum_speed_sq = 0;
float max_acc = 0;
float max_speed = 0;

float *g_sum_speed_sq, *g_max_acc, *g_max_speed;
particle_t* g_particles;

void init() {
  /* Nothing to do */
}

#ifdef DISPLAY
extern Display *theDisplay;  /* These three variables are required to open the */
extern GC theGC;             /* particle plotting window.  They are externally */
extern Window theMain;       /* declared in ui.h but are also required here.   */
#endif


//from: https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void reset_forces(particle_t* gprt, int nparticles){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(x >= nparticles) return;
  
  gprt[x].x_force = 0;
  gprt[x].y_force = 0;
}

__device__ void compute_force_gpu(particle_t * gprt, float x_pos, float y_pos, float mass){
    particle_t *ip = &gprt[x];
  float x_sep = x_pos - gprt->x_pos;
  float y_sep = y_pos - gprt->y_pos;
  float dist_sq = MAX((x_sep*x_sep) + (y_sep*y_sep), 0.01);

  /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
  float grav_base = GRAV_CONSTANT*(gprt->mass)*(mass)/dist_sq;
  ip->x_force += grav_base*x_sep;
  ip->y_force += grav_base*y_sep;
}
__global__ void compute_forces(particle_t* gprt, int nparticles){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x >= nparticles) return;
    for(int j=0; j<nparticles; j++) {
        particle_t*p = &gprt[j];
        /* compute the force of particle j on particle i */
        compute_force_gpu(&gprt[x], p->x_pos, p->y_pos, p->mass);
    }
}

__global__ void move_particles(particle_t* gprt,int nparticles, float step,float *gmacc, float *gms, float *gssq){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(x >= nparticles) return;
  particle *p = &gprt[x];
  //printf("particle %d, step %f: f(x,y), m(x)= , v(x,y)= %f, %f, %f, %f,%f\n",x,step, p->x_force, p->y_force, p->mass, p->x_vel, p->y_vel);
  p->x_pos += (p->x_vel)*step;
  p->y_pos += (p->y_vel)*step;
  float x_acc = p->x_force/p->mass;
  float y_acc = p->y_force/p->mass;
  p->x_vel += x_acc*step;
  p->y_vel += y_acc*step;
  //printf("p(%d): %f, %f\n",x, p->x_pos, p->y_pos);

  /* compute statistics */
  float cur_acc = (x_acc*x_acc + y_acc*y_acc);
  cur_acc = sqrt(cur_acc);
  float speed_sq = (p->x_vel)*(p->x_vel) + (p->y_vel)*(p->y_vel);
  float cur_speed = sqrt(speed_sq);
  //printf("stats %d: %f, %f\n",x, cur_acc, cur_speed);
  //Save statistics
  atomicMax(gms, cur_speed);
  atomicMax(gmacc, cur_acc);
  atomicAdd(gssq, speed_sq);

  //printf("global stats %d: %f, %f\n",x, *gmacc, *gms);
}

/*  Move particles one time step. Update positions, velocity, and acceleration.
    Return local computations.
*/
void all_move_particles(float step){
  //set grids and blocks
  dim3 grid(nparticles/1024+1);
  dim3 block(1024);

  //set forces of particle array to 0.
  reset_forces<<<grid, block>>>(g_particles, nparticles);
  gpuErrchk( cudaPeekAtLastError() );
  cudaDeviceSynchronize();
  //compute forces.
  compute_forces<<<grid, block>>>(g_particles, nparticles);
  gpuErrchk( cudaPeekAtLastError() );
  cudaDeviceSynchronize();
  //move each particle.
  move_particles<<<grid, block>>>(g_particles,nparticles,step,g_max_acc,g_max_speed,g_sum_speed_sq);
  gpuErrchk( cudaPeekAtLastError() );
  cudaDeviceSynchronize();
}

/* display all the particles */
void draw_all_particles() {
  int i;
  for(i=0; i<nparticles; i++) {
    int x = POS_TO_SCREEN(particles[i].x_pos);
    int y = POS_TO_SCREEN(particles[i].y_pos);
    draw_point (x,y);
  }
}

void print_all_particles(FILE* f) {
  int i;
  for(i=0; i<nparticles; i++) {
    particle_t*p = &particles[i];
    fprintf(f, "particle={pos=(%f,%f), vel=(%f,%f)}\n", p->x_pos, p->y_pos, p->x_vel, p->y_vel);
  }
}

void run_simulation() {
  float t = 0.0, dt = 0.01;
  
  while (t < T_FINAL && nparticles>0) {
    /* Update time. */
    t += dt;
    /* Move particles with the current and compute rms velocity. */
    all_move_particles(dt);

    /* Adjust dt based on maximum speed and acceleration--this
       simple rule tries to insure that no velocity will change
       by more than 10% */

    //copy statistics from device to host
    gpuErrchk(cudaMemcpy(&max_speed, g_max_speed, sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&max_acc, g_max_acc, sizeof(float), cudaMemcpyDeviceToHost));
    dt = 0.1*max_speed/max_acc;

    /* Plot the movement of the particle */
#if DISPLAY
    clear_display();
    draw_all_particles();
    flush_display();
#endif
  }
}

/*
  Simulate the movement of nparticles particles.
*/
int main(int argc, char**argv){
  if(argc >= 2)
    nparticles = atoi(argv[1]);
  if(argc == 3)
    T_FINAL = atof(argv[2]);

  init();

  /* Allocate global shared arrays for the particles data set. */
  particles = (particle_t*) malloc(sizeof(particle_t)*nparticles);
  all_init_particles(nparticles, particles);

  /* Initialize thread data structures */
  #ifdef DISPLAY
  /* Open an X window to display the particles */
  simple_init (100,100,DISPLAY_SIZE, DISPLAY_SIZE);
  #endif

  struct timeval t1, t2;
  gettimeofday(&t1, NULL);

  //Initialize CUDA
  cudaEvent_t start, stop ;
  float milliseconds = 0.0;
  cudaEventCreate(&start); cudaEventCreate(&stop) ;
  cudaEventRecord(start);

  gpuErrchk(cudaMalloc(&g_max_speed, sizeof(float)));
  gpuErrchk(cudaMemcpy(g_max_speed, &max_speed, sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&g_sum_speed_sq, sizeof(float)));
  gpuErrchk(cudaMemcpy(g_sum_speed_sq, &sum_speed_sq, sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&g_max_acc, sizeof(float)));
  gpuErrchk(cudaMemcpy(g_max_acc, &max_acc, sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&g_particles, nparticles*sizeof(particle_t)));
  gpuErrchk(cudaMemcpy(g_particles, particles, nparticles*sizeof(particle_t), cudaMemcpyHostToDevice));
  
  /* Main thread starts simulation ... */
  run_simulation();

  //Get result and free memory
  gpuErrchk(cudaMemcpy(particles, g_particles, nparticles*sizeof(particle_t), cudaMemcpyDeviceToHost));

  cudaEventRecord(stop);
  cudaEventSynchronize(stop) ; // Guarantees that the event has been executed
  cudaEventElapsedTime(&milliseconds, start, stop) ;
  cudaFree(g_particles);
  cudaFree(&g_max_speed);
  cudaFree(&g_sum_speed_sq);
  cudaFree(&g_max_acc);

  gettimeofday(&t2, NULL);

  float duration = (float) milliseconds;

#ifdef DUMP_RESULT
  FILE* f_out = fopen("particles.log", "w");
  assert(f_out);
  print_all_particles(f_out);
  fclose(f_out);
#endif

  printf("-----------------------------\n");
  printf("nparticles: %d\n", nparticles);
  printf("T_FINAL: %f\n", T_FINAL);
  printf("-----------------------------\n");
  printf("Simulation took %lf s to complete\n", duration);

#ifdef DISPLAY
  clear_display();
  draw_all_particles();
  flush_display();

  printf("Hit return to close the window.");

  getchar();
  /* Close the X window used to display the particles */
  XCloseDisplay(theDisplay);
#endif
  return 0;
}
