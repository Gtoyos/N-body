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
#include <omp.h>

#ifdef DISPLAY
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif

#include "ui.h"
#include "nbody.h"
#include "nbody_tools.h"

FILE* f_out=NULL;

int P=1,pid=0;
float *x_pos, *y_pos;
float *local_x_pos,*local_y_pos;
int delta0,delta1,local_np; //For mapping processes to particles.
int *local_nps,*displacements; //For using AllgatherV

int send_counter=0;
int msgs_to_send,msgs_to_recv;
float *send_x_forces, *send_y_forces;
float *recv_x_forces, *recv_y_forces;
MPI_Request *reqs_send, *reqs_recv;

int nparticles=10;      /* number of particles */
float T_FINAL=1.0;     /* simulation end time */
particle_t*particles;

float sum_speed_sq = 0;
float max_acc = 0;
float max_speed = 0;

void init() {
  /* Nothing to do */
}

#ifdef DISPLAY
extern Display *theDisplay;  /* These three variables are required to open the */
extern GC theGC;             /* particle plotting window.  They are externally */
extern Window theMain;       /* declared in ui.h but are also required here.   */
#endif

/* compute the force that a particle with position (x_pos, y_pos) and mass 'mass'
 * applies to particle p
 */
void compute_force(particle_t*p, int pidx, float x, float y, float mass, int j,float *xf,float*yf) {
  float x_sep, y_sep, dist_sq, grav_base;

  x_sep = x - local_x_pos[pidx-delta0];
  y_sep = y - local_y_pos[pidx-delta0];
  dist_sq = MAX((x_sep*x_sep) + (y_sep*y_sep), 0.01);

  /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
  grav_base = GRAV_CONSTANT*(p->mass)*(mass)/dist_sq;

  *xf = grav_base*x_sep;
  *yf = grav_base*y_sep;	
  p->x_force += *xf;
  p->y_force += *yf;
}

/* compute the new position/velocity */
void move_particle(particle_t*p,int pidx, float step) {

  local_x_pos[pidx-delta0] += (p->x_vel)*step;
  local_y_pos[pidx-delta0] += (p->y_vel)*step;
  float x_acc = p->x_force/p->mass;
  float y_acc = p->y_force/p->mass;
  p->x_vel += x_acc*step;
  p->y_vel += y_acc*step;

  /* compute statistics */
  float cur_acc = (x_acc*x_acc + y_acc*y_acc);
  cur_acc = sqrt(cur_acc);
  float speed_sq = (p->x_vel)*(p->x_vel) + (p->y_vel)*(p->y_vel);
  float cur_speed = sqrt(speed_sq);

  sum_speed_sq += speed_sq;
  max_acc = MAX(max_acc, cur_acc);
  max_speed = MAX(max_speed, cur_speed);
}


/*
  Move particles one time step.

  Update positions, velocity, and acceleration.
  Return local computations.
*/
void all_move_particles(float step){
  //Reset force vector
  #pragma omp parallel for schedule(static)
  for(int i=delta0; i<delta1; i++){
    particles[i].x_force = 0;
    particles[i].y_force = 0;
  }
  //Reset send force vector
  #pragma omp parallel for schedule(static)
  for(int k=0; k<msgs_to_send*send_counter;k++){
    send_x_forces[k]=0;
    send_y_forces[k]=0;
  }

  float xf=0,yf=0;
  //Comptue forces of particles in the local subset
  #pragma omp parallel for schedule(dynamic)
  for(int i=delta0; i<delta1; i++) {
    for(int j=delta0; j<i; j++) {
      particle_t*p = &particles[j];
      /* compute the force of particle j on particle i */
      compute_force(&particles[i],i,x_pos[j],y_pos[j], p->mass,j,&xf,&yf);
      p->x_force -= xf;
      p->y_force -= yf;
    }
  }
  //Compute forces of particles to send
  #pragma omp parallel for schedule(static)
  for(int i=delta0; i<delta1; i++){
    int d=0;
    for(int pnode=0; pnode<msgs_to_send;pnode++){
      for(int j=displacements[(pid+pnode+1)%P]; j<displacements[(pid+pnode+1)%P]+local_nps[(pid+pnode+1)%P]; j++){
        particle_t*p = &particles[j];
        /* compute the force of particle j on particle i */
        compute_force(&particles[i],i,x_pos[j],y_pos[j], p->mass,j,&xf,&yf);
        send_x_forces[d+j-displacements[(pid+pnode+1)%P]] -= xf;
        send_y_forces[d+j-displacements[(pid+pnode+1)%P]] -= yf;
      }
      d+=local_nps[(pid+pnode+1)%P];
    }
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

  //Add received forces
  #pragma omp parallel for schedule(static)
  for(int p=0;p<msgs_to_recv;p++){
    for(int i=delta0;i<delta1;i++){
      particles[i].x_force += recv_x_forces[p*local_np+i-delta0];
      particles[i].y_force += recv_y_forces[p*local_np+i-delta0];
    }
  }

  /* then move all particles and return statistics */
  #pragma omp parallel for schedule(static)
  for(int i=delta0; i<delta1; i++) {
    move_particle(&particles[i],i,step);
  }
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

    /* Synchronize positions*/
		MPI_Allgatherv(local_x_pos, local_np, MPI_FLOAT, x_pos, local_nps, displacements, MPI_FLOAT, MPI_COMM_WORLD);
		MPI_Allgatherv(local_y_pos, local_np, MPI_FLOAT, y_pos, local_nps, displacements, MPI_FLOAT, MPI_COMM_WORLD);

    /* Adjust dt based on maximum speed and acceleration--this
       simple rule tries to insure that no velocity will change
       by more than 10%. Share this info between all processes*/
    MPI_Allreduce(MPI_IN_PLACE, &max_speed, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_acc, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
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
  particles = malloc(sizeof(particle_t)*nparticles);
  all_init_particles(nparticles, particles); //NOTE: We can execute this function in parallel as it is deterministic.

  /* create arrays to share for particle position */
  x_pos = malloc(sizeof(float)*nparticles);
  y_pos = malloc(sizeof(float)*nparticles);
  /* copy positions*/
  for(int i=0; i<nparticles;i++){
    x_pos[i] = particles[i].x_pos;
    y_pos[i] = particles[i].y_pos;
  }

	/*Data structures for MPI Synchronization*/
	local_nps = malloc(sizeof(int)*P);
	displacements = malloc(sizeof(int)*P);
	delta0 = pid*(nparticles/P);
	delta1 = (pid==P-1)? nparticles : delta0+(nparticles/P);
	local_np = delta1-delta0;
  MPI_Allgather(&local_np, 1, MPI_INT, local_nps, 1, MPI_INT, MPI_COMM_WORLD); // Share how many particles each process has.
  MPI_Allgather(&delta0, 1, MPI_INT, displacements, 1, MPI_INT, MPI_COMM_WORLD); // Share from what index each process starts.

	/*Create arrays for storing local particles temporarily*/
  local_x_pos = malloc(sizeof(float)*local_np);
  local_y_pos = malloc(sizeof(float)*local_np);
	for(int i=delta0; i<delta1; i++){
		local_x_pos[i-delta0] = x_pos[i];
		local_y_pos[i-delta0] = y_pos[i];
	}

  /* Create arrays for sending and receiving forces*/
  msgs_to_send = (P%2==1) ? (P-1)/2 : ((pid>=P/2)? (P/2)-1 : P/2);
  msgs_to_recv = msgs_to_send + ((P%2==0)?((pid>=P/2)?1:-1):0);
  for(int i=0;i<msgs_to_send;i++)
    send_counter+=local_nps[(pid+i+1)%P];
  send_x_forces = malloc(sizeof(float)*send_counter);
  send_y_forces = malloc(sizeof(float)*send_counter);
  recv_x_forces = malloc(sizeof(float)*msgs_to_recv*local_np);
  recv_y_forces = malloc(sizeof(float)*msgs_to_recv*local_np);
  reqs_send = malloc(sizeof(MPI_Request)*msgs_to_send*2);
  reqs_recv = malloc(sizeof(MPI_Request)*msgs_to_recv*2);
  for(int k=0; k<send_counter;k++){
    send_x_forces[k]=0;
    send_y_forces[k]=0;
  }
  for(int k=0; k<msgs_to_recv*local_np;k++){
    recv_x_forces[k]=0;
    recv_y_forces[k]=0;
  }

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
	MPI_Finalize();
  return 0;
}