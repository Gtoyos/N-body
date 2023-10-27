CC	= gcc
NVCC = nvcc
NVFLAGS	= -O0 -g -arch=sm_50 -lX11 -I/netfs/inf/trahay_f/mpich/include/mpi -I/netfs/inf/trahay_f/mpich/include
CFLAGS = -O0 -g -lX11
LDFLAGS = -lm -lpthread -lX11 -lcudart -lcublas  -L/netfs/inf/trahay_f/mpich/lib -lmpi_cxx -lmpi -lopen-rte -lopen-pal -ldl -lnsl -lutil -lm 
VERBOSE	=
TARGET	= nbody_brute_force nbody_barnes_hut nbody_brute_force_v3
OBJS	= ui.o xstuff.o nbody_tools.o nbody_alloc.o

#DISPLAY = -DDISPLAY
DISPLAY =

DUMP = -DDUMP_RESULT
#DUMP =

all: $(TARGET)

nbody_brute_force: nbody_brute_force.o $(OBJS)
	$(NVCC) $(VERBOSE) -o $@ $< $(OBJS) $(LDFLAGS)

nbody_barnes_hut: nbody_barnes_hut.o $(OBJS)
	$(NVCC) $(VERBOSE) -o $@ $< $(OBJS)  $(LDFLAGS)

nbody_brute_force_v3: nbody_brute_force_v3.o $(OBJS)
	$(NVCC) $(VERBOSE) -o $@ $< $(OBJS) $(LDFLAGS)
	
%.o: %.c
	$(NVCC) $(CFLAGS) -c $< $(VERBOSE) $(DISPLAY) $(DUMP)

%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< $(VERBOSE) $(DISPLAY) $(DUMP)

clean:
	rm -f *.o $(TARGET)
