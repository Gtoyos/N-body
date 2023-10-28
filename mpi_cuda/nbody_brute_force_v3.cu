/*
** nbody_brute_force.c - nbody simulation using the brute-force algorithm (O(n*n))
** PARALLEL MPI VERSION
** EACH PROCESS GETS A SUBSET OF PARTICLES TO SIMULATE - POSITION SYNCHRONIZATION IS DONE USING MPI_ALLGATHERV
** 
** 
**/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <unistd.h>
#include <mpi.h>
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

FILE* f_out=NULL;

int P=1,pid=0;
float *x_pos, *y_pos;
float *local_x_pos,*local_y_pos;
int delta0,delta1,local_np; //For mapping processes to particles.
int *local_nps,*displacements, *disp_part; //For using AllgatherV

int send_counter=0;
int msgs_to_send,msgs_to_recv;
float *send_x_forces, *send_y_forces;
float *recv_x_forces, *recv_y_forces;
MPI_Request *reqs_send, *reqs_recv;

float *g_sum_speed_sq, *g_max_acc, *g_max_speed;
particle_t *g_particles;
float *g_x_pos, *g_y_pos, *g_local_x_pos, *g_local_y_pos;
float *g_send_x_forces, *g_send_y_forces, *g_recv_x_forces, *g_recv_y_forces;
int *g_local_nps, *g_displacements, *g_disp_part;

int nparticles=10;      /* number of particles */
float T_FINAL=1.0;     /* simulation end time */
particle_t*particles;

float sum_speed_sq = 0;
float max_acc = 0;
float max_speed = 0;

void init() {
  /* Nothing to do */
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

__global__ void reset_forces(particle_t* gprt, int local_np, int delta0){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(x >= local_np) return;
  
  gprt[delta0+x].x_force = 0;
  gprt[delta0+x].y_force = 0;
}

__global__ void reset_send_forces(float*g_send_x_forces,float *g_send_y_forces,int send_counter){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(x >= send_counter) return;
  g_send_x_forces[x] = 0;
  g_send_y_forces[x] = 0;
}

__global__ void compute_forces_local(particle_t* gprt,float* g_x_pos,float *g_y_pos,int local_np,int delta0,int nparticles){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= local_np) return;
  if(y >= local_np) return;
  particle_t *ip = &gprt[delta0+y]; //forces i will save
  particle_t *jp = &gprt[delta0+x];

  float x_sep = g_x_pos[delta0+x] - g_x_pos[delta0+y];
  float y_sep = g_y_pos[delta0+x] - g_y_pos[delta0+y];
  float dist_sq = MAX((x_sep*x_sep) + (y_sep*y_sep), 0.01);

  /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
  float grav_base = GRAV_CONSTANT*(ip->mass)*(jp->mass)/dist_sq;
  //printf("f(%d,%d): %f, %f\n",delta0+x,delta0+y, grav_base*x_sep, grav_base*y_sep);
  atomicAdd(&(ip->x_force), grav_base*x_sep);
  atomicAdd(&(ip->y_force), grav_base*y_sep);  
}

__global__ void compute_forces_external(particle_t* gprt,float* g_x_pos,float *g_y_pos,float * g_send_x_forces, float * g_send_y_forces,int local_np,int delta0,int msgs_to_send,int * g_local_nps,int *g_displacements,int pid, int P, int *g_disp_part){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if(x >= local_np) return;
  if(z >= msgs_to_send) return;
  if(y >= g_local_nps[(pid+z+1)%P]) return;

  particle_t *ip = &gprt[delta0+x]; //forces i will save
  particle_t *jp = &gprt[g_displacements[(pid+z+1)%P]+y]; //forces j will save

  float x_sep = g_x_pos[g_displacements[(pid+z+1)%P]+y] - g_x_pos[delta0+x];
  float y_sep = g_y_pos[g_displacements[(pid+z+1)%P]+y] - g_y_pos[delta0+x];
  float dist_sq = MAX((x_sep*x_sep) + (y_sep*y_sep), 0.01);

  /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
  float grav_base = GRAV_CONSTANT*(ip->mass)*(jp->mass)/dist_sq;
  //printf("f(%d,%d): %f, %f\n",delta0+x,g_displacements[(pid+z+1)%P]+y, grav_base*x_sep, grav_base*y_sep);
  float xf=grav_base*x_sep;
  float yf=grav_base*y_sep;
  atomicAdd(&(ip->x_force), xf);
  atomicAdd(&(ip->y_force), yf);
  atomicAdd(&g_send_x_forces[g_disp_part[z]+y], -xf);
  atomicAdd(&g_send_y_forces[g_disp_part[z]+y], -yf);  
}

__global__ void move_particles(particle_t* gprt,float *g_local_x_pos, float *g_local_y_pos, int local_np, int delta0, float step,float *gmacc, float *gms, float *gssq){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(x >= local_np) return;
  particle *p = &gprt[delta0+x];
  //printf("particle %d, step %f: f(x,y), m(x)= , v(x,y)= %f, %f, %f, %f,%f\n",x,step, p->x_force, p->y_force, p->mass, p->x_vel, p->y_vel);
  g_local_x_pos[x] += (p->x_vel)*step;
  g_local_y_pos[x] += (p->y_vel)*step;
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

__global__ void add_rcv_forces(particle_t* gprt,float *g_recv_x_forces,float *g_recv_y_forces,int local_np,int delta0, int msgs_to_recv){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= local_np) return;
  if(y >= msgs_to_recv) return;
  particle_t *p = &gprt[delta0+x];
  atomicAdd(&(p->x_force), g_recv_x_forces[y*local_np+x]);
  atomicAdd(&(p->y_force), g_recv_y_forces[y*local_np+x]);
}

/*
  Move particles one time step.

  Update positions, velocity, and acceleration.
  Return local computations.
*/
void all_move_particles(float step){
  //set grids and blocks
  dim3 grid(local_np/1024+1);
  dim3 block(1024);
  dim3 grid2(local_np/32+1, local_np/32+1);
  dim3 block2(32,32);
  dim3 grid3(local_np/32+1,(local_np*2)/32+1,msgs_to_send);
  dim3 block3(32,32,1);
  dim3 grid4(local_np/1024+1,msgs_to_recv);
  dim3 block4(1024,1);
  
  //set forces of particle array to 0.
  reset_forces<<<grid, block>>>(g_particles, local_np,delta0);
  gpuErrchk( cudaPeekAtLastError() );
  cudaDeviceSynchronize();

  reset_send_forces<<<send_counter/1024+1, block>>>(g_send_x_forces, g_send_y_forces, send_counter);
  gpuErrchk( cudaPeekAtLastError() );
  cudaDeviceSynchronize();

  //Comptue forces of particles in the local subset
  compute_forces_local<<<grid2, block2>>>(g_particles,g_x_pos,g_y_pos,local_np,delta0,nparticles);
  gpuErrchk( cudaPeekAtLastError() );
  cudaDeviceSynchronize();

  if(msgs_to_send>0){
    //Compute forces of particles to send
    compute_forces_external<<<grid3,block3>>>(g_particles,g_x_pos,g_y_pos,g_send_x_forces,g_send_y_forces,local_np,delta0,
      msgs_to_send,g_local_nps,g_displacements,pid,P,g_disp_part);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();

    //Copy send force array back to host
    gpuErrchk(cudaMemcpy(send_x_forces, g_send_x_forces, send_counter*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(send_y_forces, g_send_y_forces, send_counter*sizeof(float), cudaMemcpyDeviceToHost));
  }
  /* Synchronize force computation*/
  int d=0;
  for(int i=0;i<msgs_to_send;i++){
    MPI_Isend(send_x_forces+d,local_nps[(pid+i+1)%P],MPI_FLOAT,(pid+i+1)%P,0,MPI_COMM_WORLD,reqs_send+i);
    MPI_Isend(send_y_forces+d,local_nps[(pid+i+1)%P],MPI_FLOAT,(pid+i+1)%P,0,MPI_COMM_WORLD,reqs_send+i+msgs_to_send);
    d+=local_nps[(pid+i+1)%P];
  }
  for(int i=0;i<msgs_to_recv;i++){
    MPI_Irecv(recv_x_forces+i*local_np,local_np,MPI_FLOAT,(P+pid-i-1)%P,0,MPI_COMM_WORLD,reqs_recv+i);
    MPI_Irecv(recv_y_forces+i*local_np,local_np,MPI_FLOAT,(P+pid-i-1)%P,0,MPI_COMM_WORLD,reqs_recv+i+msgs_to_recv);
  }
  MPI_Waitall(2*msgs_to_send,reqs_send,MPI_STATUSES_IGNORE);
  MPI_Waitall(2*msgs_to_recv,reqs_recv,MPI_STATUSES_IGNORE);

  if(msgs_to_recv>0){
    //Copy received forces to device
    gpuErrchk(cudaMemcpy(g_recv_x_forces, recv_x_forces, msgs_to_recv*local_np*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(g_recv_y_forces, recv_y_forces, msgs_to_recv*local_np*sizeof(float), cudaMemcpyHostToDevice));

    //Add received forces
    add_rcv_forces<<<grid4,block4>>>(g_particles,g_recv_x_forces,g_recv_y_forces,local_np,delta0,msgs_to_recv);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();
  }
  /* then move all particles and return statistics */
  move_particles<<<grid, block>>>(g_particles,g_local_x_pos,g_local_y_pos,local_np,delta0,step,g_max_acc,g_max_speed,g_sum_speed_sq);
  gpuErrchk( cudaPeekAtLastError() );
  cudaDeviceSynchronize();
}

/* display all the particles */
void draw_all_particles() {
  for(int i=0; i<nparticles; i++) {
    int x = POS_TO_SCREEN(x_pos[i]);
    int y = POS_TO_SCREEN(y_pos[i]);
    draw_point (x,y);
  }
}

void print_all_particles(FILE* f) {
  for(int i=0; i<nparticles; i++) {
    particle_t*p = &particles[i];
    fprintf(f, "particle={pos=(%f,%f), vel=(%f,%f)}\n", x_pos[i], y_pos[i], p->x_vel, p->y_vel);
  }
}

void run_simulation() {
  float t = 0.0, dt = 0.01;
  while (t < T_FINAL && nparticles>0) {
    /* Update time. */
    t += dt;
    /* Move particles with the current and compute rms velocity. */
    all_move_particles(dt);
    //printf("--%f--\n",t);
    /* Copy local positions to host*/
    gpuErrchk(cudaMemcpy(local_x_pos, g_local_x_pos, local_np*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(local_y_pos, g_local_y_pos, local_np*sizeof(float), cudaMemcpyDeviceToHost));

    /* Synchronize positions*/
		MPI_Allgatherv(local_x_pos, local_np, MPI_FLOAT, x_pos, local_nps, displacements, MPI_FLOAT, MPI_COMM_WORLD);
		MPI_Allgatherv(local_y_pos, local_np, MPI_FLOAT, y_pos, local_nps, displacements, MPI_FLOAT, MPI_COMM_WORLD);

    /* Send updated positions to device*/
    gpuErrchk(cudaMemcpy(g_x_pos, x_pos, nparticles*sizeof(float), cudaMemcpyHostToDevice));	
    gpuErrchk(cudaMemcpy(g_y_pos, y_pos, nparticles*sizeof(float), cudaMemcpyHostToDevice));

    //copy statistics from device to host
    gpuErrchk(cudaMemcpy(&max_speed, g_max_speed, sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&max_acc, g_max_acc, sizeof(float), cudaMemcpyDeviceToHost));
    /* Adjust dt based on maximum speed and acceleration--this
       simple rule tries to insure that no velocity will change
       by more than 10%. Share this info between all processes*/
    MPI_Allreduce(MPI_IN_PLACE, &max_speed, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_acc, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

    //send stats to device
    gpuErrchk(cudaMemcpy(g_max_speed, &max_speed, sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(g_max_acc, &max_acc, sizeof(float), cudaMemcpyHostToDevice));
    dt = 0.1*max_speed/max_acc;

    #if DISPLAY
    /* Plot the movement of the particle */
    clear_display();
    draw_all_particles();
    flush_display();
    #endif
  }
}


//Simulate the movement of nparticles particles.
int main(int argc, char**argv){
  if(argc >= 2)
    nparticles = atoi(argv[1]);
  if(argc == 3)
    T_FINAL = atof(argv[2]);

  init();
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);

  /* Allocate global shared arrays for the particles data set. */
  particles = (particle_t*)malloc(sizeof(particle_t)*nparticles);
  all_init_particles(nparticles, particles); //NOTE: We can execute this function in parallel as it is deterministic.

  /* create arrays to share for particle position */
  x_pos = (float *)malloc(sizeof(float)*nparticles);
  y_pos = (float *)malloc(sizeof(float)*nparticles);
  /* copy positions*/
  for(int i=0; i<nparticles;i++){
    x_pos[i] = particles[i].x_pos;
    y_pos[i] = particles[i].y_pos;
  }

	/*Data structures for MPI Synchronization*/
	local_nps = (int*)malloc(sizeof(int)*P);
	displacements = (int*)malloc(sizeof(int)*P);
	delta0 = pid*(nparticles/P);
	delta1 = (pid==P-1)? nparticles : delta0+(nparticles/P);
	local_np = delta1-delta0;
  MPI_Allgather(&local_np, 1, MPI_INT, local_nps, 1, MPI_INT, MPI_COMM_WORLD); // Share how many particles each process has.
  MPI_Allgather(&delta0, 1, MPI_INT, displacements, 1, MPI_INT, MPI_COMM_WORLD); // Share from what index each process starts.

	/*Create arrays for storing local particles temporarily*/
  local_x_pos = (float *)malloc(sizeof(float)*local_np);
  local_y_pos = (float *)malloc(sizeof(float)*local_np);
	for(int i=delta0; i<delta1; i++){
		local_x_pos[i-delta0] = x_pos[i];
		local_y_pos[i-delta0] = y_pos[i];
	}

  /* Create arrays for sending and receiving forces*/
  msgs_to_send = (P%2==1) ? (P-1)/2 : ((pid>=P/2)? (P/2)-1 : P/2);
  msgs_to_recv = msgs_to_send + ((P%2==0)?((pid>=P/2)?1:-1):0);
  for(int i=0;i<msgs_to_send;i++)
    send_counter+=local_nps[(pid+i+1)%P];

  send_x_forces = (float *)malloc(sizeof(float)*send_counter);
  send_y_forces = (float *) malloc(sizeof(float)*send_counter);
  recv_x_forces = (float *) malloc(sizeof(float)*msgs_to_recv*local_np);
  recv_y_forces = (float *) malloc(sizeof(float)*msgs_to_recv*local_np);
  reqs_send = (MPI_Request*)malloc(sizeof(MPI_Request)*msgs_to_send*2);
  reqs_recv = (MPI_Request*)malloc(sizeof(MPI_Request)*msgs_to_recv*2);
  for(int k=0; k<send_counter;k++){
    send_x_forces[k]=0;
    send_y_forces[k]=0;
  }
  for(int k=0; k<msgs_to_recv*local_np;k++){
    recv_x_forces[k]=0;
    recv_y_forces[k]=0;
  }
  disp_part = (int*)malloc(sizeof(int)*msgs_to_send);
  disp_part[0]= 0;
  for(int i=1;i<msgs_to_send;i++)
    disp_part[i] = disp_part[i-1]+local_nps[(pid+i)%P];
  
  //Start cuda arrays
  gpuErrchk(cudaMalloc(&g_particles, nparticles*sizeof(particle_t)));
  gpuErrchk(cudaMemcpy(g_particles, particles, nparticles*sizeof(particle_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&g_max_speed, sizeof(float)));
  gpuErrchk(cudaMemcpy(g_max_speed, &max_speed, sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&g_sum_speed_sq, sizeof(float)));
  gpuErrchk(cudaMemcpy(g_sum_speed_sq, &sum_speed_sq, sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&g_max_acc, sizeof(float)));
  gpuErrchk(cudaMemcpy(g_max_acc, &max_acc, sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&g_x_pos, nparticles*sizeof(float)));
  gpuErrchk(cudaMemcpy(g_x_pos, x_pos, nparticles*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&g_y_pos, nparticles*sizeof(float)));
  gpuErrchk(cudaMemcpy(g_y_pos, y_pos, nparticles*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&g_local_x_pos, local_np*sizeof(float)));
  gpuErrchk(cudaMemcpy(g_local_x_pos, local_x_pos, local_np*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&g_local_y_pos, local_np*sizeof(float)));
  gpuErrchk(cudaMemcpy(g_local_y_pos, local_y_pos, local_np*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&g_send_x_forces, send_counter*sizeof(float)));
  gpuErrchk(cudaMemcpy(g_send_x_forces, send_x_forces, send_counter*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&g_send_y_forces, send_counter*sizeof(float)));
  gpuErrchk(cudaMemcpy(g_send_y_forces, send_y_forces, send_counter*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&g_recv_x_forces, msgs_to_recv*local_np*sizeof(float)));
  gpuErrchk(cudaMemcpy(g_recv_x_forces, recv_x_forces, msgs_to_recv*local_np*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&g_recv_y_forces, msgs_to_recv*local_np*sizeof(float)));
  gpuErrchk(cudaMemcpy(g_recv_y_forces, recv_y_forces, msgs_to_recv*local_np*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&g_local_nps, P*sizeof(int)));
  gpuErrchk(cudaMemcpy(g_local_nps, local_nps, P*sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&g_displacements, P*sizeof(int)));
  gpuErrchk(cudaMemcpy(g_displacements, displacements, P*sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&g_disp_part, msgs_to_send*sizeof(int)));
  gpuErrchk(cudaMemcpy(g_disp_part, disp_part, msgs_to_send*sizeof(int), cudaMemcpyHostToDevice));

	/* Initialize thread data structures */
	#ifdef DISPLAY
	if(pid==0){
		/* Open an X window to display the particles */
		simple_init (100,100,DISPLAY_SIZE, DISPLAY_SIZE);
	}
	#endif

  struct timeval t1, t2;
  gettimeofday(&t1, NULL);

  /* START SIMULATION */
  run_simulation();

  gettimeofday(&t2, NULL);

  float duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

  #ifdef DUMP_RESULT
  if(pid==0){
    FILE* f_out = fopen("particles.log", "w");
    assert(f_out);
    print_all_particles(f_out);
    fclose(f_out);
  }
  #endif
  if(pid==0){
    printf("-----------------------------\n");
    printf("nparticles: %d\n", nparticles);
    printf("T_FINAL: %f\n", T_FINAL);
    printf("-----------------------------\n");
    printf("Simulation took %lf s to complete\n", duration);
  }
  #ifdef DISPLAY
  if(pid==0){
    clear_display();
    draw_all_particles();
    flush_display();

    printf("Hit return to close the window.");

    getchar();
    /* Close the X window used to display the particles */
    XCloseDisplay(theDisplay);
  }
  #endif
	free(x_pos);
	free(y_pos);
	free(local_x_pos);
	free(local_y_pos);
  free(recv_x_forces);
  free(recv_y_forces);
  free(send_x_forces);
  free(send_y_forces);
  free(reqs_send);
  free(reqs_recv);
  free(disp_part);

  cudaFree(g_particles);
  cudaFree(g_max_speed);
  cudaFree(g_sum_speed_sq);
  cudaFree(g_max_acc);
  cudaFree(g_x_pos);
  cudaFree(g_y_pos);
  cudaFree(g_local_x_pos);
  cudaFree(g_local_y_pos);
  cudaFree(g_send_x_forces);
  cudaFree(g_send_y_forces);
  cudaFree(g_recv_x_forces);
  cudaFree(g_recv_y_forces);
  cudaFree(g_local_nps);
  cudaFree(g_displacements);
  cudaFree(g_disp_part);

	MPI_Finalize();
  return 0;
}