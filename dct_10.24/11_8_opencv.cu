//#include<iostream>
//#include<fstream>
//#include<math.h>
//#include<string>
//#include<stdio.h>
//#include<cuda_runtime.h>
//#include<device_launch_parameters.h>
//using namespace std;

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

#include <cmath>
#include <cstdio>

#include <opencv2/core/core.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>

//using namespace std;
using std::ifstream;
using std::string;
using std::cout;
using std::endl;
using std::ios;
using std::setiosflags;
using std::setprecision;

//#define length 8
#define PI 3.14159265
#define length 256
#define block_len 4

cudaError_t dctWithCuda_1(const double *d, double *D);

cudaError_t dctWithCuda_2(const double *f, double *F);

void dct(double *f, double *F){
	int i,j,t;
	//double data[length]={0.0};
	double tmp;

	double data[length] = {0.0};
	for(t=0; t<length; t++)
	{
		for (i=0; i<length; i++)
				data[i] = f[t*length+i];//load row data from f.

		for(i=0; i<length; i++)
		{
			if(i==0)
			{
				tmp = (double)(1.0/sqrt(1.0*length));
				F[t*length+i] = 0.0;//why use F[bid]? Do transpose at the same time.
				for(j=0; j<length; j++)
					F[t*length+i] +=data[j] ;
				F[t*length] *= tmp;
			}
			else
			{
				tmp = (double)(sqrt(2.0/(1.0*length)));
				for(i=1; i<length; i++){
					F[t*length+i] = 0;
					for(j=0; j<length; j++)
						F[t*length+i] += (double)(data[j]*cos((2*j+1)*i*PI/(2.0*length)));
					F[t*length+i] *= tmp;
				}
			}
		}
	}

	for(t=0; t<length; t++)
	{
		for(i=0; i<length; i++)
			data[i] = F[i*length+t];
		for(i=0; i<length; i++)
		{
			if(i==0)
			{
				tmp=(double)(1.0/sqrt(1.0*length));
				F[t]=0;
				for(j=0; j<length; j++)
					F[t] += data[j];
				F[t] *= tmp;
			}
			else
			{
				tmp = (double)(sqrt(2.0/(1.0*length)));
				for(i=1; i<length; i++)
				{
					F[i*length+t] = 0;
					for(j=0; j<length; j++)
						F[i*length+t] += (double)(data[j]*cos((2*j+1)*i*PI/(2*length)));
					F[i*length+t] *= tmp;
				}
			}
		}
	}
}

__global__ void dct_1(const double *f,double *F){
	int bid = blockIdx.x;
	//int tid = threadIdx.x;
	int i,j;
	//double data[length]={0.0};
	double tmp;
	//printf("");
	if(bid<length){
		double data[length];
		for (i=0; i<length; i++)
			data[i] = f[bid*length+i];//load row data from f.
		__syncthreads();
		for(i=0; i<length; i++){
			if(i==0){
				tmp = (double)(1.0/sqrt(1.0*length));
				F[bid * length + i] = 0.0;
				for(j=0; j<length; j++)
					F[bid*length+i] +=data[j] ;
				F[bid*length] *= tmp;
			}
			else{
				tmp = (double)(sqrt(2.0/(1.0*length)));
				for(i=1; i<length; i++){
					F[bid*length+i] = 0;
					for(j=0; j<length; j++)
						F[bid*length+i] += (double)(data[j]*cos((2*j+1)*i*PI/(2*length)));
					F[bid*length+i] *= tmp;
				}
			}
		}
		__syncthreads();
		for(i=0; i<length; i++)
			data[i] = F[i*length+bid];
		__syncthreads();
		for(i=0; i<length; i++){
			if(i==0){
				tmp=(double)(1.0/sqrt(1.0*length));
				F[bid]=0;
				for(j=0; j<length; j++)
					F[bid] += data[j];
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
	if(tidx<length && tidy<length)
	{
		for(i=0; i<length*length; i++)
		{
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
	double f[length*length] = {0,0};
	double F0[length*length] = {0.0};
	double F1[length*length] = {0.0};
	double F2[length*length] = {0.0};
	while(i<length*length){
		if(getline(infile, line))
		{
			f[i] = atof(line.c_str());
		}
		i++;
	}
//	cout<<"before"<<endl;
//	for(i=0; i<length*length; i++){
//			cout<<f[i]<<" ";
//			if ((i+1)%length==0)
//				cout<<endl;
//		}
//	cout<<endl;
//	for(i=0; i<length*length; i++){
//			cout<<F1[i]<<" ";
//			if ((i+1)%length==0)
//					cout<<endl;
//	}

	clock_t star0, end0;
	clock_t star1, end1;
	clock_t star2, end2;

	//use event to record time
	//float time0 = 0;
	float time1 = 0;
	float time2 = 0;
//	cudaEvent_t start0, stop0;
//	cudaEventCreate(&start0);
//	cudaEventCreate(&stop0);

	cudaEvent_t start1, stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);

	cudaEvent_t start2, stop2;
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);

	/*
	 * excute dct()
	 */
	star0 = clock();
	dct(f,F0);
	end0 = clock();

//	cout<<"----------------dct()-----------"<<endl;
//	for (i = 0; i < length * length; i++)
//	{
//		 /* GPU can't calculate floating number precisely
//		 * 0 will be a very small floating number.
//		 * so print this numbers with 7 digits after decimal point
//		*/
//		cout << setiosflags(ios::right)
//			<< setiosflags(ios::fixed) << setprecision(7)
//			<< F0[i] << "\t";
//		if ((i + 1) % length == 0)
//			cout << endl;
//	}


	/*
	 * excute dct_1()
	 */
	star1 = clock();
	cudaEventRecord(start1, 0 );
	cudaError_t cudaStatus = dctWithCuda_1(f,F1);
	if (cudaStatus != cudaSuccess)
	{
	        fprintf(stderr, "dctWithCuda_1 failed!");
	        return 1;
	}
	end1 = clock();
	cudaEventRecord(stop1, 0 );

	cudaEventSynchronize(start1);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&time1,start1,stop1);

	printf("excute1 time: %f (ms)\n",time1);
	cudaEventDestroy(start1);    //destory the event
	cudaEventDestroy(stop1);

//	cout<<"----------------dct_1()-----------"<<endl;
//	for (i = 0; i < length * length; i++)
//	{
//		 /* GPU can't calculate floating number precisely
//		 * 0 will be a very small floating number.
//		 * so print this numbers with 7 digits after decimal point
//		*/
//		cout << setiosflags(ios::right)
//			<< setiosflags(ios::fixed) << setprecision(7)
//			<< F1[i] << "\t";
//		if ((i + 1) % length == 0)
//			cout << endl;
//	}


	/*
	 * excute dct_2()
	 */
	star2 = clock();
	cudaEventRecord(start2, 0 );
	cudaError_t cudaStatus_ = dctWithCuda_2(f,F2);
	if (cudaStatus_ != cudaSuccess)
	{
			fprintf(stderr, "dctWithCuda_1 failed!");
		    return 1;
	}
	cudaEventRecord(stop2, 0 );
	end2 = clock();

	cudaEventSynchronize(start2);
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&time2,start2,stop2);

	printf("excute2 time: %f (ms)\n",time2);
	cudaEventDestroy(start2);    //destory the event
	cudaEventDestroy(stop2);

//	for (i = 0; i < length * length; i++)
//	{
//		 /* GPU can't calculate floating number precisely
//		 * 0 will be a very small floating number.
//		 * so print this numbers with 7 digits after decimal point
//		*/
//		cout << setiosflags(ios::right)
//			<< setiosflags(ios::fixed) << setprecision(7)
//			<< F2[i] << "\t";
//		if ((i + 1) % length == 0)
//			cout << endl;
//	}


	//time
	cout<<"----------------clock()-----------"<<endl;
	cout<< "dct() timeused="<<end0-star0<<"ms"<<endl;
	cout<< "dct_1() timeused="<<end1-star1<<"ms"<<endl;
	cout<< "dct_2() timeused="<<end2-star2<<"ms"<<endl;

	double mat_cv[8][8]={7.24224225e-02, 7.83669204e-03, 1.07341856e-01, 1.71479255e-01, 3.10998634e-02, 1.29764974e-02, 2.41287440e-01, 4.43069041e-02,
			1.13842189e-02, 6.47377223e-02, 6.94301277e-02, 2.13208988e-01,2.00988576e-01,1.29413247e-01, 1.38687521e-01, 3.10612798e-01,
			1.67800680e-01, 1.66601896e-01,9.28850099e-02, 1.21374808e-01, 2.71115124e-01, 2.14537248e-01, 2.14956731e-01, 2.92603731e-01,
			3.65073085e-01, 2.95452982e-01, 3.42667162e-01, -1.79812834e-02, -1.95148848e-02, 5.05965427e-02, 4.69370112e-02, -3.16086859e-02,
			-8.29299539e-03, 5.44460639e-02, -1.06319003e-02, 5.43169677e-04, -3.70726548e-02, -3.47166322e-02, 3.31790783e-02, 4.27808166e-02,
			-3.20185833e-02, -1.41328480e-02, 5.69829792e-02, -5.29655814e-03, -4.77588549e-03, 2.39680521e-03, -2.20994279e-03, 5.39673865e-02,
			7.19442889e-02, 1.80451013e-03, 7.75165297e-03, 6.01153113e-02, 9.75808874e-03, 1.15207210e-02, 5.55298850e-02, -6.58077225e-02,
			2.46187001e-02, 6.02829829e-02, -5.40484488e-03, 4.01977524e-02, 8.33794400e-02, -3.48810852e-03, 5.85473813e-02, 6.00872934e-02};




	return 0;

}

cudaError_t dctWithCuda_1(const double *d, double *D){
	double *dev_d = 0;
	double *dev_D = 0;
	float time=0;
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


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0 );
    cudaStatus = cudaMemcpy(D, dev_D, length*length* sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
    cudaEventRecord(stop, 0 );
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time,start,stop);

    printf("copy1 time: %f (ms)\n",time);

Error:
	cudaFree(dev_d);
	cudaFree(dev_D);
	cudaEventDestroy(start);    //destory the event
	cudaEventDestroy(stop);
	return cudaStatus;
}


cudaError_t dctWithCuda_2(const double *d, double *D){
	double *dev_d = 0;
	double *dev_D = 0;
	float time=0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_d,length * length * sizeof(double));
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_D,length * length * sizeof(double));
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_d, d,length * length * sizeof(double),cudaMemcpyHostToDevice);
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0 );
    cudaStatus = cudaMemcpy(D, dev_D, length*length * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
    cudaEventRecord(stop, 0 );
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time,start,stop);

    printf("copy2 time: %f (ms)\n",time);
Error:
	cudaFree(dev_d);
	cudaFree(dev_D);
	cudaEventDestroy(start);    //destory the event
	cudaEventDestroy(stop);
	return cudaStatus;
}















