#include<iostream>
#include<fstream>
#include<math.h>
#include<string>
#include<stdio.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
using namespace std;

//#define length 8
#define PI 3.14159265
#define length 8
#define block_len 16

cudaError_t dctWithCuda_1(const double *d, double *D);

cudaError_t dctWithCuda_2(const double *f, double *F);

/*__global__ void dct1(float *f, float *F){
	int tidy = blockIdx.x*blockDim.x + threadIdx.x;
	int tidx = blockIdx.y*blockDim.y + threadIdx.y;
	int index = tidx*len + tidy;
	float tmp;
	float beta,alfa;
	if(tidx == 0)
		beta = sqrt(1.0/length);
	else
		beta = sqrt(2.0/length);
	if(tidy == 0)
		alfa = sqrt(1.0/length);
	else
		alfa = sqrt(2.0/length);
	if(tidx<length && tidy<length){
		for(i=0; i<length; i++){
			int x = i/length;
			int y = i%length;
			tmp+=((int)data[i])*cos((2*x+1)*tidx*PI/(2.0*length))*
					cos((2*y+1)*tidy*PI/(2.0*length));
		}
		F[index]=(float)alfa*beta*tmp;
	}
}*/

__global__ void dct_1(const double *f,double *F){
	int bid = blockIdx.x;
	//int tid = threadIdx.x;
	int i,j;
	//double data[length]={0.0};
	double tmp;
	if(bid<length){
		__shared__ double data[length];
		for (i=0; i<length; i++)
			data[i] = f[bid*length+i];//load row data from f.
		__syncthreads();
		for(i=0; i<length; i++){
			if(i==0){
				tmp = (double)(1.0/sqrt(1.0*length));
				F[bid] = 0;//why use F[bid]? Do transpose at the same time.
				for(j=0; j<length; j++)
					F[bid] +=data[j] ;
				F[bid] *= tmp;
			}
			else{
				tmp = (double)(sqrt(2.0/(1.0*length)));
				for(i=1; i<length; i++){
					F[i*length+bid] = 0;
					for(j=0; j<length; j++)
						F[i*length+bid] += (double)(data[j]*cos((2*j+1)*i*PI/(2*length)));
					F[i*length+bid] *= tmp;
				}
			}
		}
		__syncthreads();
		for(i=0; i<length; i++)
			data[i] = F[bid*length+1];
		__syncthreads();
		for(i=0; i<length; i++){
			if(i==0){
				tmp=(double)(1.0/sqrt(1.0*length));
				F[bid]=0;
				for(j=0; j<length; j++)
					F[bid] += data[i];
				F[bid] *= tmp;
			}
			else{
				tmp = (double)(sqrt(2.0/(1.0*length)));
				for(i=1; i<length; i++){
					F[i*length+bid] = 0;
					for(j=0; j<length; j++)
						F[i*length+bid] += (double)(data[j]*cos((2*j+1)*i*PI/(2*length)));
					F[i*length+bid] *= tmp;
				}
			}
		}
		__syncthreads();
	}
}

__global__ void dct_2(const double *f, double *F){
	int tidy = blockIdx.x*blockDim.x + threadIdx.x;
	int tidx = blockIdx.y*blockDim.y + threadIdx.y;
	int index = tidx*length + tidy;
	int i;
	double tmp;
	double beta ,alfa;
	if(tidx == 0)
		beta = sqrt(1.0/length);
	else
		beta = sqrt(2.0/length);
	if(tidy == 0)
		alfa = sqrt(1.0/length);
	else
		alfa = sqrt(2.0/length);
	if(tidx<length && tidy<length){
		for(i=0; i<length*length; i++){
			int x = i/length;
			int y = i%length;
			tmp += ((double)f[i])*cos((2*x+1)*tidx*PI/(2.0*length))*
					cos((2*y+1)*tidy*PI/(2.0*length));
		}
		F[index]=(double)alfa * beta * tmp;
	}
}

int main(){
	ifstream infile("/home/zhujian/cuda-workspace/dct_10.16/gradient.txt");
	int i=0;
	string line;
	double f[length*length] = {0.0};
	double F[length*length] = {0.0};
	while(i<length*length){
		if(getline(infile, line)){

			f[i] = atof(line.c_str());
			cout<<"f[i]:  "<<f[i]<<endl;
		}
		i++;
	}
	cout<<"before"<<endl;
	for(i=0; i<length*length; i++){
			cout<<f[i]<<" ";
			if ((i+1)%length==0)
				cout<<endl;
		}
	cout<<endl;
	for(i=0; i<length*length; i++){
			cout<<F[i]<<" ";
			if ((i+1)%length==0)
					cout<<endl;
	}
	cudaError_t cudaStatus = dctWithCuda_1(f,F);
	if (cudaStatus != cudaSuccess) {
	        fprintf(stderr, "dctWithCuda_1 failed!");
	        return 1;
	}
	cout<<"after"<<endl;
	for(i=0; i<length*length; i++){
		cout<<f[i]<<" ";
		if ((i+1)%length==0)
			cout<<endl;
	}
	cout<<endl;
	for(i=0; i<length*length; i++){
			cout<<F[i]<<" ";
			if ((i+1)%length==0)
					cout<<endl;
	}
	return 0;

}

cudaError_t dctWithCuda_1(const double *d, double *D){
	double *dev_d = 0;
	double *dev_D = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_d,length *length* sizeof(double));
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_D,length *length* sizeof(double));
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_d, d,length *length*sizeof(double),cudaMemcpyHostToDevice);
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMemcpy-- failed");
		goto Error;
	}
	//launch a kernel on the GPU
	dct_1<<<length,1>>>(dev_d, dev_D);

	cudaStatus = cudaThreadSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaThreadSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(D, dev_D, length*length* sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
Error:
	cudaFree(dev_d);
	cudaFree(dev_D);
	return cudaStatus;
}


cudaError_t dctWithCuda_2(const double *d, double *D){
	double *dev_d = 0;
	double *dev_D = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_d,length * sizeof(double));
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_D,length * sizeof(double));
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_d, d,length *sizeof(double),cudaMemcpyHostToDevice);
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMalloc failed");
		goto Error;
	}

	//launch a kernel on the GPU
	dct_2<<<1, (length/block_len)*(length/block_len), block_len*block_len>>>(dev_d, dev_D);

	cudaStatus = cudaThreadSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaThreadSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(D, dev_D, length*length * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
Error:
	cudaFree(dev_d);
	cudaFree(dev_D);

	return cudaStatus;
}
















