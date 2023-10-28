#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <mpi.h>

#ifdef DISPLAY
#include <X11/Xlib.h>
#include "ui.h"
#endif

#include "nbody.h"
#include "nbody_tools.h"

FILE* f_out = NULL;

int nparticles = 10;      /* number of particles */
float T_FINAL = 1.0;     /* simulation end time */
particle_t* particles;
node_t* root;

float sum_speed_sq = 0;
float max_acc = 0;
float max_speed = 0;

int P=1,pid=0;
float *x_f, *y_f;
float *local_x_f,*local_y_f;
int delta0,delta1,local_np; //For mapping processes to particles.
int *local_nps,*displacements; //For using AllgatherV


void init() {
  init_alloc(8 * nparticles);
  root = malloc(sizeof(node_t));
  init_node(root, NULL, XMIN, XMAX, YMIN, YMAX);
}

#ifdef DISPLAY
extern Display *theDisplay;
extern GC theGC;
extern Window theMain;
#endif

void compute_force(particle_t* p, int part_idx,float x_pos, float y_pos, float mass) {
  float x_sep, y_sep, dist_sq, grav_base;

  x_sep = x_pos - p->x_pos;
  y_sep = y_pos - p->y_pos;
  dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);

  grav_base = GRAV_CONSTANT * (p->mass) * (mass) / dist_sq;

  local_x_f[part_idx] += grav_base * x_sep;
  local_y_f[part_idx] += grav_base * y_sep;
}

void compute_force_on_particle(node_t* n, particle_t* p, int part_idx) {
  if (!n || n->n_particles == 0) {
    return;
  }
  if (n->particle) {
    assert(n->children == NULL);
    compute_force(p,part_idx, n->x_center, n->y_center, n->mass);
  } else {
#define THRESHOLD 2
    float size = n->x_max - n->x_min;
    float diff_x = n->x_center - p->x_pos;
    float diff_y = n->y_center - p->y_pos;
    float distance = sqrt(diff_x * diff_x + diff_y * diff_y);

    if (size / distance < THRESHOLD) {
      compute_force(p,part_idx, n->x_center, n->y_center, n->mass);
    } else {
      int i;
      for (i = 0; i < 4; i++) {
        compute_force_on_particle(&n->children[i], p,part_idx);
      }
    }
  }
}

void move_particle(particle_t* p, float step, node_t* new_root) {
  assert(p->node != NULL);
  p->x_pos += (p->x_vel) * step;
  p->y_pos += (p->y_vel) * step;
  float x_acc = p->x_force / p->mass;
  float y_acc = p->y_force / p->mass;
  p->x_vel += x_acc * step;
  p->y_vel += y_acc * step;

  float cur_acc = (x_acc * x_acc + y_acc * y_acc);
  cur_acc = sqrt(cur_acc);
  float speed_sq = (p->x_vel) * (p->x_vel) + (p->y_vel) * (p->y_vel);
  float cur_speed = sqrt(speed_sq);

  sum_speed_sq += speed_sq;
  max_acc = MAX(max_acc, cur_acc);
  max_speed = MAX(max_speed, cur_speed);

  p->node = NULL;
  if (p->x_pos < new_root->x_min || p->x_pos > new_root->x_max || p->y_pos < new_root->y_min || p->y_pos > new_root->y_max) {
    nparticles--;
  } else {
    insert_particle(p, new_root);
  }
}

void move_particles_in_node(node_t* n, float step, node_t* new_root) {
  if (!n) return;

  if (n->particle) {
    particle_t* p = n->particle;
    move_particle(p, step, new_root);
  }
  if (n->children) {
    int i;
    for (i = 0; i < 4; i++) {
      move_particles_in_node(&n->children[i], step, new_root);
    }
  }
}

void all_compute_forces() {
  for (int i = 0; i < local_np; i++) {
    local_x_f[i] = 0;
    local_y_f[i] = 0;
    compute_force_on_particle(root, &particles[i+delta0],i);
  }
}

void all_move_particles(float step) {
  //compute_force_in_node(root);

  node_t* new_root = alloc_node();
  init_node(new_root, NULL, XMIN, XMAX, YMIN, YMAX);

  move_particles_in_node(root, step, new_root);

  free_node(root);
  root = new_root;
}

void run_simulation() {
  float t = 0.0, dt = 0.01;

  while (t < T_FINAL && nparticles > 0) {
    t += dt;

    //Compute local forces
    all_compute_forces();
    
    //Synchronize forces
    MPI_Allgatherv(local_x_f, local_np, MPI_FLOAT, x_f, local_nps, displacements, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(local_y_f, local_np, MPI_FLOAT, y_f, local_nps, displacements, MPI_FLOAT, MPI_COMM_WORLD);
    
    //Update forces
    for (int i = 0; i < nparticles; i++) {
      particles[i].x_force = x_f[i];
      particles[i].y_force = y_f[i];
    }

    //Update quadtree
    all_move_particles(dt);

    dt = 0.1 * max_speed / max_acc;

#if DISPLAY
    if(pid==0){
      node_t* n = root;
      clear_display();
      draw_node(n);
      flush_display();
    }
#endif
  }
}

void insert_all_particles(int nparticles, particle_t* particles, node_t* root) {
  int i;
  for (i = 0; i < nparticles; i++) {
    insert_particle(&particles[i], root);
  }
}

int main(int argc, char** argv) {
  if (argc >= 2) {
    nparticles = atoi(argv[1]);
  }
  if (argc == 3) {
    T_FINAL = atof(argv[2]);
  }

  init();
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);

  particles = malloc(sizeof(particle_t) * nparticles);
  all_init_particles(nparticles, particles);
  insert_all_particles(nparticles, particles, root);

  /* create arrays to share forces */
  x_f = malloc(sizeof(float)*nparticles);
  y_f = malloc(sizeof(float)*nparticles);
  /* copy positions*/
  for(int i=0; i<nparticles;i++){
    x_f[i] = 0;
    y_f[i] = 0;
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
  local_x_f = malloc(sizeof(float)*local_np);
  local_y_f = malloc(sizeof(float)*local_np);
	for(int i=delta0; i<delta1; i++){
		local_x_f[i-delta0] = 0;
		local_y_f[i-delta0] = 0;
	}

#ifdef DISPLAY
  simple_init(100, 100, DISPLAY_SIZE, DISPLAY_SIZE);
#endif

  struct timeval t1, t2;
  gettimeofday(&t1, NULL);

  run_simulation();

  gettimeofday(&t2, NULL);

  float duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
  
  #ifdef DUMP_RESULT
  if(pid==0){
    FILE* f_out = fopen("particles.log", "w");
    assert(f_out);
    print_particles(f_out, root);
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
	free(displacements);
	free(local_nps);
  free(local_x_f);
  free(local_y_f);
	free(y_f);
	free(x_f);
	printf("Process %d finished\n", pid);
	MPI_Finalize();
  return 0;
}
