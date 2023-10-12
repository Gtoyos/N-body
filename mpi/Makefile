#Mahcines: 3a401-13 -> 24 cores / 48 threads
#          b313-13 -> 16 cores / 32 threads

CC	= mpicc
CFLAGS	= -O0 -g -Wall
LDFLAGS = -lm -lpthread -lX11
VERBOSE	=
TARGET	= nbody_brute_force nbody_barnes_hut nbody_brute_force_v2
OBJS	= ui.o xstuff.o nbody_tools.o nbody_alloc.o

#DISPLAY = -DDISPLAY
DISPLAY =

DUMP = -DDUMP_RESULT
#DUMP =

all: $(TARGET)

nbody_brute_force: nbody_brute_force.o $(OBJS)
	$(CC) $(VERBOSE) -o $@ $< $(OBJS) $(LDFLAGS)

nbody_brute_force_v2: nbody_brute_force_v2.o $(OBJS)
	$(CC) $(VERBOSE) -o $@ $< $(OBJS) $(LDFLAGS)

nbody_barnes_hut: nbody_barnes_hut.o $(OBJS)
	$(CC) $(VERBOSE) -o $@ $< $(OBJS)  $(LDFLAGS)

test: test.o $(OBJS)
	$(CC) $(VERBOSE) -o test $< $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< $(VERBOSE) $(DISPLAY) $(DUMP)
clean:
	rm -f *.o $(TARGET)
