/*
** nbody_brute_force.c - nbody simulation using the brute-force algorithm (O(n*n))
** NOTE: Implementation Using FP Precision.
** Note: Version where we have sepparate arrays for each particle attribute.
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
float *g_particles_xforce, *g_particles_yforce, *g_particles_mass, *g_particles_xvel, *g_particles_yvel, *g_particles_xpos, *g_particles_ypos;
float *particles_xforce, *particles_yforce, *particles_mass, *particles_xvel, *particles_yvel, *particles_xpos, *particles_ypos;
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

__global__ void reset_forces(float * fx,float *fy, int nparticles){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(x >= nparticles) return;
  
  fx[x] = 0;
  fy[x] = 0;
}

__global__ void compute_forces(float *m, float*fx, float*fy, float *px, float *py,int nparticles){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= nparticles) return;
  if(y >= nparticles) return;
  //if(x>y) return;

  float x_sep = px[y] - px[x];
  float y_sep = py[y] - py[x];
  float dist_sq = MAX((x_sep*x_sep) + (y_sep*y_sep), 0.01);

  /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
  float grav_base = GRAV_CONSTANT*(m[x])*(m[y])/dist_sq;
  //printf("f(%d,%d): %f, %f\n",x,y, grav_base*x_sep, grav_base*y_sep);
  atomicAdd(&fx[x], grav_base*x_sep);
  atomicAdd(&fy[x], grav_base*y_sep);
  //atomicAdd(&fx[y], -grav_base*x_sep);
  //atomicAdd(&fy[y], -grav_base*y_sep);    
}

__global__ void move_particles(float*m,float*fx,float*fy,float*vx,float*vy,float*px,float*py,int nparticles, float step,float *gmacc, float *gms, float *gssq){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(x >= nparticles) return;
  //printf("particle %d, step %f: f(x,y), m(x)= , v(x,y)= %f, %f, %f, %f,%f\n",x,step, fx[x], fy[x], m[x], vx[x], vy[x]);
  px[x] += vx[x]*step;
  py[x] += vy[x]*step;
  float x_acc = fx[x]/m[x];
  float y_acc = fy[x]/m[x];
  vx[x] += x_acc*step;
  vy[x] += y_acc*step;
  //printf("p(%d): %f, %f\n",x, p->x_pos, p->y_pos);

  /* compute statistics */
  float cur_acc = (x_acc*x_acc + y_acc*y_acc);
  cur_acc = sqrt(cur_acc);
  float speed_sq = (vx[x])*(vx[x]) + (vy[x])*(vy[x]);
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
  dim3 grid2(nparticles/32+1, nparticles/32+1);
  dim3 block2(32,32);

  //set forces of particle array to 0.
  reset_forces<<<grid, block>>>(g_particles_xforce, g_particles_yforce, nparticles);
  gpuErrchk( cudaPeekAtLastError() );
  cudaDeviceSynchronize();
  //compute forces.
  compute_forces<<<grid2, block2>>>(g_particles_mass, g_particles_xforce, g_particles_yforce, g_particles_xpos, g_particles_ypos, nparticles);
  gpuErrchk( cudaPeekAtLastError() );
  cudaDeviceSynchronize();
  //move each particle.
  move_particles<<<grid, block>>>(g_particles_mass,g_particles_xforce,g_particles_yforce,g_particles_xvel,g_particles_yvel,g_particles_xpos,g_particles_ypos,nparticles,step,g_max_acc,g_max_speed,g_sum_speed_sq);
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
    fprintf(f, "particle={pos=(%f,%f), vel=(%f,%f)}\n", particles_xpos[i], particles_ypos[i], particles_xvel[i], particles_yvel[i]);
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

  //split particles into separate arrays
  particles_xforce = (float*) malloc(sizeof(float)*nparticles);
  particles_yforce = (float*) malloc(sizeof(float)*nparticles);
  particles_mass = (float*) malloc(sizeof(float)*nparticles);
  particles_xvel = (float*) malloc(sizeof(float)*nparticles);
  particles_yvel = (float*) malloc(sizeof(float)*nparticles);
  particles_xpos = (float*) malloc(sizeof(float)*nparticles);
  particles_ypos = (float*) malloc(sizeof(float)*nparticles);

  //copy data from particles to separate arrays
  for(int i=0; i<nparticles; i++){
    particles_xforce[i] = particles[i].x_force;
    particles_yforce[i] = particles[i].y_force;
    particles_mass[i] = particles[i].mass;
    particles_xvel[i] = particles[i].x_vel;
    particles_yvel[i] = particles[i].y_vel;
    particles_xpos[i] = particles[i].x_pos;
    particles_ypos[i] = particles[i].y_pos;
  }

  //print particle info
  //printf("nparticles: %d\n", nparticles);
  //for(int i=0; i<nparticles; i++){
  //  printf("particle %d: mass=%f, xforce=%f, yforce=%f, xvel=%f, yvel=%f, xpos=%f, ypos=%f\n",i, particles_mass[i], particles_xforce[i], particles_yforce[i], particles_xvel[i], particles_yvel[i], particles_xpos[i], particles_ypos[i]);
 //}

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

  //create separate particle arrays;
  gpuErrchk(cudaMalloc(&g_particles_mass, nparticles*sizeof(float)));
  gpuErrchk(cudaMalloc(&g_particles_xforce, nparticles*sizeof(float)));
  gpuErrchk(cudaMalloc(&g_particles_yforce, nparticles*sizeof(float)));
  gpuErrchk(cudaMalloc(&g_particles_xvel, nparticles*sizeof(float)));
  gpuErrchk(cudaMalloc(&g_particles_yvel, nparticles*sizeof(float)));
  gpuErrchk(cudaMalloc(&g_particles_xpos, nparticles*sizeof(float)));
  gpuErrchk(cudaMalloc(&g_particles_ypos, nparticles*sizeof(float)));

  //copy data from host to device
  gpuErrchk(cudaMemcpy(g_particles_mass, particles_mass, nparticles*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(g_particles_xforce, particles_xforce, nparticles*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(g_particles_yforce, particles_yforce, nparticles*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(g_particles_xvel, particles_xvel, nparticles*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(g_particles_yvel, particles_yvel, nparticles*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(g_particles_xpos, particles_xpos, nparticles*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(g_particles_ypos, particles_ypos, nparticles*sizeof(float), cudaMemcpyHostToDevice));

  /* Main thread starts simulation ... */
  run_simulation();

  //Get result and free memory
  gpuErrchk(cudaMemcpy(particles_mass, g_particles_mass, nparticles*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(particles_xforce, g_particles_xforce, nparticles*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(particles_yforce, g_particles_yforce, nparticles*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(particles_xvel, g_particles_xvel, nparticles*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(particles_yvel, g_particles_yvel, nparticles*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(particles_xpos, g_particles_xpos, nparticles*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(particles_ypos, g_particles_ypos, nparticles*sizeof(float), cudaMemcpyDeviceToHost));


  cudaEventRecord(stop);
  cudaEventSynchronize(stop) ; // Guarantees that the event has been executed
  cudaEventElapsedTime(&milliseconds, start, stop) ;

  cudaFree(g_max_speed);
  cudaFree(g_sum_speed_sq);
  cudaFree(g_max_acc);
  cudaFree(g_particles_mass);
  cudaFree(g_particles_xforce);
  cudaFree(g_particles_yforce);
  cudaFree(g_particles_xvel);
  cudaFree(g_particles_yvel);
  cudaFree(g_particles_xpos);
  cudaFree(g_particles_ypos);
  
  gettimeofday(&t2, NULL);

  float duration = (float) milliseconds/1000;

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
