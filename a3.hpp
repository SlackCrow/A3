#ifndef A3_HPP
#define A3_HPP

__global__ 
void gaussianKDEKernel(float x_inp, int x_loc, int n, float h, float *in, float *out) 
{
    extern __shared__  float tempSave[];
    int currentThreadId = threadIdx.x;
    int id = (blockDim.x * blockDim.x) + currentThreadId;
    if(currentThreadId == 0) tempSave[0] = 0; // ensuring that atomicAdd doesn't result in an undefined behavior if other threads get it done faster than tid 0

    float innerX = (x_inp - in[id]) / h;
    float k = (1/sqrtf(2*PIVALUE)) * expf(-(powf(innerX,2)/2)); // avoids double precision, GPUs usually have 1/4-1/32 ratio for FP64 

    tempSave[currentThreadId] = k;
    atomicAdd(tempSave,tempSave[currentThreadId]); // add K when i == currentThreadId to tempSave[0], faster if it runs on CC > 7.0(Volta)
    __syncthreads(); // wait until all threads are done with atomicAdd
    	
    if(currentThreadId == 0) out[x_loc] += (tempSave[0] * (1/(n*h)));
}

void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) 
{
    float *x_in;
    float *y_out;

    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared); // 96KB per SM(with Volta and Turing), configured to 80KB for shared memory and 16KB for L1 cache
    cudaMalloc((void **)&x_in, n*sizeof(float));
    cudaMallocManaged((void **)&y_out, n*sizeof(float)); // Used managed for output as it is only used when the kernel saves an output, works fine after Pascal, a bit slower before

    cudaMemcpy(x_in, &x[0], n*sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t cStream[n]; // create a stream for each i of n

    for (int i = 0; i<n; i++) 
    {
        cudaStreamCreateWithFlags(&cStream[i],cudaStreamNonBlocking); // set the stream to non blocking
    	gaussianKDEKernel <<<(n+255)/256, 256, 256*sizeof(float),cStream[i]>>> (x[i], i, n, h, x_in, y_out);
    }

    cudaDeviceSynchronize(); // synch everything
	cudaFree(x_in);
    
    y.assign(y_out,y_out+n);
} // gaussian_kde

#endif // A3_HPP
