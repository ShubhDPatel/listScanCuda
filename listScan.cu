/* ACADEMIC INTEGRITY PLEDGE                                              */
/*                                                                        */
/* - I have not used source code obtained from another student nor        */
/*   any other unauthorized source, either modified or unmodified.        */
/*                                                                        */
/* - All source code and documentation used in my program is either       */
/*   my original work or was derived by me from the source code           */
/*   published in the textbook for this course or presented in            */
/*   class.                                                               */
/*                                                                        */
/* - I have not discussed coding details about this project with          */
/*   anyone other than my instructor. I understand that I may discuss     */
/*   the concepts of this program with other students and that another    */
/*   student may help me debug my program so long as neither of us        */
/*   writes anything during the discussion or modifies any computer       */
/*   file during the discussion.                                          */
/*                                                                        */
/* - I have violated neither the spirit nor letter of these restrictions. */
/*                                                                        */
/*                                                                        */
/*                                                                        */
/* Signed: Shubh Patel  Date: 4/6/2024                                    */
/*                                                                        */
/*                                                                        */
/* CPSC 677 CUDA Prefix Sum lab, Version 1.02, Spring 2024.               */

#include "helper_timer.h"
#include <stdio.h>
#include <stdlib.h>

// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// + lst[n-1]}

#define BLOCK_SIZE 512 //@@ You can change this

__global__ void scan(int* input, int* output, int* aux, int len)
{
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here

    // Initialize the shared memory
    extern __shared__ float temp[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len)
    {
        temp[2 * threadIdx.x] = input[i];
        // temp[2 * threadIdx.x + 1] = input[i];
    }

    for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride <<= 1)
    {
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < 2 * BLOCK_SIZE)
        {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < 2 * BLOCK_SIZE)
        {
            temp[index + stride] += temp[index];
        }
    }
    __syncthreads();
    if (i < len)
    {
        output[i] = temp[2 * threadIdx.x];
        aux[i] = temp[2 * threadIdx.x + 1];
    }
}

int main(int argc, char** argv)
{
    int* hostInput; // The input 1D list
    int* hostOutput; // The output list
    int* expectedOutput;
    int* deviceInput;
    int* deviceOutput;
    int *deviceAuxArray, *deviceAuxScannedArray;
    int numElements; // number of elements in the list

    FILE *infile, *outfile;
    int inputLength, outputLength;
    StopWatchLinux stw;
    unsigned int blog = 1;

    // Import host input data
    stw.start();
    if ((infile = fopen("input.raw", "r")) == NULL)
    {
        printf("Cannot open input.raw.\n");
        exit(EXIT_FAILURE);
    }
    fscanf(infile, "%i", &inputLength);
    hostInput = (int*)malloc(sizeof(int) * inputLength);
    for (int i = 0; i < inputLength; i++)
        fscanf(infile, "%i", &hostInput[i]);
    fclose(infile);
    numElements = inputLength;
    hostOutput = (int*)malloc(numElements * sizeof(int));
    stw.stop();
    printf("Importing data and creating memory on host: %f ms\n", stw.getTime());

    if (blog)
        printf("*** The number of input elements in the input is %i\n", numElements);

    stw.reset();
    stw.start();

    cudaMalloc((void**)&deviceInput, numElements * sizeof(int));
    cudaMalloc((void**)&deviceOutput, numElements * sizeof(int));

    cudaMalloc(&deviceAuxArray, (BLOCK_SIZE << 1) * sizeof(int));
    cudaMalloc(&deviceAuxScannedArray, (BLOCK_SIZE << 1) * sizeof(int));

    stw.stop();
    printf("Allocating GPU memory: %f ms\n", stw.getTime());

    stw.reset();
    stw.start();

    cudaMemset(deviceOutput, 0, numElements * sizeof(int));

    stw.stop();
    printf("Clearing output memory: %f ms\n", stw.getTime());

    stw.reset();
    stw.start();

    cudaMemcpy(deviceInput, hostInput, numElements * sizeof(int),
        cudaMemcpyHostToDevice);

    stw.stop();
    printf("Copying input memory to the GPU: %f ms\n", stw.getTime());

    //@@ Initialize the grid and block dimensions here
    dim3 blockDim(BLOCK_SIZE, 1, 1);
    dim3 gridDim(ceil((float)numElements / blockDim.x), 1, 1);

    stw.reset();
    stw.start();

    //@@ Modify this to complete the functionality of the scan
    //@@ on the device
    scan<<<gridDim, blockDim, 2 * BLOCK_SIZE * sizeof(float)>>>(deviceInput, deviceOutput, deviceAuxArray, BLOCK_SIZE);

    cudaDeviceSynchronize();

    stw.stop();
    printf("Performing CUDA computation: %f ms\n", stw.getTime());

    stw.reset();
    stw.start();

    cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(int),
        cudaMemcpyDeviceToHost);

    stw.stop();
    printf("Copying output memory to the CPU: %f ms\n", stw.getTime());

    stw.reset();
    stw.start();

    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(deviceAuxArray);
    cudaFree(deviceAuxScannedArray);

    stw.stop();
    printf("Freeing GPU Memory: %f ms\n", stw.getTime());

    if ((outfile = fopen("output.raw", "r")) == NULL)
    {
        printf("Cannot open output.raw.\n");
        exit(EXIT_FAILURE);
    }
    fscanf(outfile, "%i", &outputLength);
    expectedOutput = (int*)malloc(sizeof(int) * outputLength);
    for (int i = 0; i < outputLength; i++)
        fscanf(outfile, "%i", &expectedOutput[i]);
    fclose(outfile);

    int test = 1;
    for (int i = 0; i < outputLength; i++)
    {
        if (expectedOutput[i] != hostOutput[i])
            printf("%i %i %i\n", i, expectedOutput[i], hostOutput[i]);
        test = test && (expectedOutput[i] == hostOutput[i]);
    }

    if (test)
        printf("Results correct.\n");
    else
        printf("Results incorrect.\n");

    free(hostInput);
    cudaFreeHost(hostOutput);
    free(expectedOutput);

    return 0;
}
