#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //Will will try an asymmetric send/recv over a buffer
    int * send = malloc(sizeof(int)*4);
    int * recv = malloc(sizeof(int)*10);
    
    //Initialize the send buffer
    for(int i = 0; i < 4; i++) {
        send[i] = i;
    }
    //Initialize the recv buffer
    for(int i = 0; i < 10; i++) {
        recv[i] = -1;
    }

    if(rank==1){
        //Send message MPI
        MPI_Send(send+2, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if(rank==0){
        MPI_Recv(recv+1,2,MPI_INT,1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    if(rank==0){
        for(int i = 0; i < 10; i++) {
            printf("%d ", recv[i]);
        }
        printf("\n");
    }
    free(send);
    free(recv);
    MPI_Finalize();

    return 0;
}