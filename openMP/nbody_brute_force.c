#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>

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

double sum_speed_sq = 0;
double max_acc = 0;
double max_speed = 0;

void init() {
  /* Nothing to do */
}

#ifdef DISPLAY
extern Display *theDisplay;
extern GC theGC;
extern Window theMain;
#endif

void compute_force(particle_t* p, double x_pos, double y_pos, double mass) {
  double x_sep, y_sep, dist_sq, grav_base;

  x_sep = x_pos - p->x_pos;
  y_sep = y_pos - p->y_pos;
  dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);

  grav_base = GRAV_CONSTANT * (p->mass) * (mass) / dist_sq;

  p->x_force += grav_base * x_sep;
  p->y_force += grav_base * y_sep;
}

void move_particle(particle_t* p, double step) {
  p->x_pos += (p->x_vel) * step;
  p->y_pos += (p->y_vel) * step;
  double x_acc = p->x_force / p->mass;
  double y_acc = p->y_force / p->mass;
  p->x_vel += x_acc * step;
  p->y_vel += y_acc * step;

  double cur_acc = (x_acc * x_acc + y_acc * y_acc);
  cur_acc = sqrt(cur_acc);
  double speed_sq = (p->x_vel) * (p->x_vel) + (p->y_vel) * (p->y_vel);
  double cur_speed = sqrt(speed_sq);

  sum_speed_sq += speed_sq;
  max_acc = MAX(max_acc, cur_acc);
  max_speed = MAX(max_speed, cur_speed);
}

void all_compute_forces(int start, int end) {
#pragma omp parallel for
  for (int i = start; i < end; i++) {
    particles[i].x_force = 0;
    particles[i].y_force = 0;
    for (int j = 0; j < nparticles; j++) {
      if (i != j) {
        particle_t* p = &particles[j];
        compute_force(&particles[i], p->x_pos, p->y_pos, p->mass);
      }
    }
  }
}

void all_move_particles(double step, int start, int end) {
#pragma omp parallel for
  for (int i = start; i < end; i++) {
    move_particle(&particles[i], step);
  }
}

void draw_all_particles() {
  int i;
  for (i = 0; i < nparticles; i++) {
    int x = POS_TO_SCREEN(particles[i].x_pos);
    int y = POS_TO_SCREEN(particles[i].y_pos);
    draw_point(x, y);
  }
}

void print_all_particles(FILE* f) {
  int i;
  for (i = 0; i < nparticles; i++) {
    particle_t* p = &particles[i];
    fprintf(f, "particle={pos=(%f,%f), vel=(%f,%f)}\n", p->x_pos, p->y_pos, p->x_vel, p->y_vel);
  }
}

void run_simulation() {
  double t = 0.0, dt = 0.01;
  while (t < T_FINAL && nparticles > 0) {
    t += dt;

    all_compute_forces(0, nparticles);
    all_move_particles(dt, 0, nparticles);

    dt = 0.1 * max_speed / max_acc;

#if DISPLAY
    clear_display();
    draw_all_particles();
    flush_display();
#endif
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

  particles = malloc(sizeof(particle_t) * nparticles);
  all_init_particles(nparticles, particles);

#ifdef DISPLAY
  simple_init(100, 100, DISPLAY_SIZE, DISPLAY_SIZE);
#endif

  struct timeval t1, t2;
  gettimeofday(&t1, NULL);

  run_simulation();

  gettimeofday(&t2, NULL);
  double duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

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
  XCloseDisplay(theDisplay);
#endif

  return 0;
}

