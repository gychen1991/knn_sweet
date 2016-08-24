#include "cublas.h"
#include<omp.h>
#include<curand.h>
#include<curand_kernel.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <pthread.h>
#include "common8.h"




__device__ __forceinline__ int F2I( float floatVal ) {
 int intVal = __float_as_int( floatVal );
 return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;
}
__device__ __forceinline__ float I2F( int intVal ) {
 return __int_as_float( (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF);
}
__device__ float atomicMin_float(float *address, float val){
    int val_int = F2I(val);
    int old = atomicMin((int *)address, val_int);
    return I2F(old);
}
__device__ float atomicMax_float(float *address, float val){
    int val_int = F2I(val);
    int old = atomicMax((int *)address, val_int);
    return I2F(old);
}
__device__ float atomicAdd_float(float *address, float val){
    int val_int = F2I(val);
    int old = atomicAdd((int *)address, val_int);
    return I2F(old);
}

void check(cudaError_t status, const char *message){
	if(status != cudaSuccess)
		cout <<message<<endl;
}

__global__ void Norm(float *point, float *norm, int size, int dim){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < size){
		float dist = 0.0f;
		for(int i = 0; i < dim; i++){
			float tmp = point[tid * dim + i];
			dist += tmp * tmp ;
		}
		norm[tid] = dist; 
	}
}

__global__ void AddAll(float *queryNorm_dev, float *repNorm_dev, float *query2reps_dev, int size, int rep_nb){
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int ty = threadIdx.y + blockIdx.y * blockDim.y;
	if(tx < size && ty < rep_nb){
		float temp = query2reps_dev[ty * size + tx];
		temp += (queryNorm_dev[tx] + repNorm_dev[ty]);
		query2reps_dev[ty * size + tx] = sqrt(temp);
	}
}
__global__ void findQCluster(float *query2reps_dev,P2R *q2rep_dev, int size, int rep_nb, float *maxquery_dev, R2all_static_dev *req2q_static_dev){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < size){
		float temp = FLT_MAX;
		int index = -1;
		for(int i = 0; i < rep_nb; i++){
			float tmp = query2reps_dev[i * size+ tid];
			if(temp > tmp){
				index = i;
				temp = tmp;
			}
		}
		q2rep_dev[tid] = {index, temp};
		atomicAdd(&req2q_static_dev[index].npoints,1);
		atomicMax_float(&maxquery_dev[index],temp);
	}
}
__global__ void findTCluster(float *source2reps_dev,P2R *s2rep_dev, int size, int rep_nb, R2all_static_dev *req2s_static_dev){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < size){
		float temp = FLT_MAX;
		int index = -1;
		for(int i = 0; i < rep_nb; i++){
			float tmp = source2reps_dev[i * size+ tid];
			if(temp > tmp){
				index = i;
				temp = tmp;
			}
		}
		s2rep_dev[tid] = {index, temp};
		atomicAdd(&req2s_static_dev[index].npoints,1);
		//atomicMax_float(&maxquery_dev[index],temp);
	}
}
__global__ void fillQMembers(P2R *q2rep_dev, int size, int *repsID, R2all_dyn_p *req2q_dyn_p_dev){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < size){
		int repId = q2rep_dev[tid].repIndex;
		int memberId = atomicAdd(&repsID[repId], 1);
		req2q_dyn_p_dev[repId].memberID[memberId] = tid;
	}
}
__global__ void fillTMembers(P2R *s2rep_dev, int size, int *repsID, R2all_dyn_p *req2s_dyn_p_dev){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < size){
		int repId = s2rep_dev[tid].repIndex;
		int memberId = atomicAdd(&repsID[repId], 1);
		req2s_dyn_p_dev[repId].sortedmembers[memberId] = {tid, s2rep_dev[tid].dist2rep};
	}
}
__device__ int reorder = 0;
__global__ void reorderMembers( int rep_nb, int *repsID, int *reorder_members,R2all_dyn_p *req2q_dyn_p_dev){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < rep_nb){
		if(repsID[tid]!=0){
			int reorderId = atomicAdd(&reorder, repsID[tid]);
		//printf("reorder Id %d %d\n",tid, repsID[tid]);//reorderId);
			memcpy(reorder_members + reorderId, req2q_dyn_p_dev[tid].memberID, repsID[tid]*sizeof(int)) ;
		}
	}
}
__global__ void selectReps_cuda(float * queries_dev, int query_nb, float *qreps_dev, int qrep_nb, int *qIndex_dev, int *totalSum_dev, int totalTest, int dim){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < totalTest * qrep_nb * qrep_nb){
		int test = tid/(qrep_nb*qrep_nb);
		int repId = int (tid%(qrep_nb*qrep_nb))/qrep_nb;
		float distance = Edistance_128(queries_dev + qIndex_dev[test * qrep_nb + repId]*dim, queries_dev + qIndex_dev[test*qrep_nb + int (tid%(qrep_nb*qrep_nb))%qrep_nb]*dim, dim);
		atomicAdd(&totalSum_dev[test],int(distance));
	}
}
__device__ int repTest = 0;
__global__ void selectReps_max(float *queries_dev, int query_nb, float *qreps_dev, int qrep_nb, int *qIndex_dev, int *totalSum_dev, int totalTest, int dim){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid == 0){
		float distance = 0.0f;
		for(int i = 0; i < totalTest; i++){
			if(distance < totalSum_dev[i]){
				//printf("distnace %d\n",totalSum_dev[i]);
				distance = totalSum_dev[i];
				repTest = i;
			}	
		}
		printf("repTest %d\n",repTest);
	}
}
__global__ void selectReps_copy(float *queries_dev, int query_nb, float *qreps_dev, int qrep_nb, int *qIndex_dev, int *totalSum_dev, int totalTest, int dim){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < qrep_nb){
		memcpy(qreps_dev+tid*dim, queries_dev + qIndex_dev[repTest * qrep_nb + tid] * dim, dim * sizeof(float));
	}
}
void print_last_error()
/* just run cudaGetLastError() and print the error message if its return value is not cudaSuccess */
{
  cudaError_t cudaError;

  cudaError = cudaGetLastError();
  if(cudaError != cudaSuccess)
  {
    printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
  }
}
void clusterReps(float * &queries_dev, float * &sources_dev, float * &qreps_dev, float * &sreps_dev, float * &maxquery_dev, \
	P2R * &q2rep_dev, P2R * &s2rep_dev, R2all_static_dev * &rep2q_static_dev, R2all_static_dev * &rep2s_static_dev, R2all_dyn_p * &rep2q_dyn_p_dev, R2all_dyn_p * &rep2s_dyn_p_dev, float * &query2reps_dev,
	P2R * &q2rep,     P2R * &s2rep,     R2all_static * &rep2q_static,         R2all_static * &rep2s_static,     R2all_dyn_v * &rep2q_dyn_v,		R2all_dyn_v * &rep2s_dyn_v,     float * &query2reps,
	R2all_dyn_p * &rep2q_dyn_p, R2all_dyn_p * &rep2s_dyn_p, int * &reorder_members){


	cudaMalloc((void **)&query2reps_dev, qrep_nb * query_nb * sizeof(float));
	cudaError_t status;
	status = cudaMalloc((void **)&queries_dev, query_nb * dim * sizeof(float));
	check(status,"Malloc queries failed\n");
	status = cudaMemcpy(queries_dev, queries, query_nb * dim * sizeof(float), cudaMemcpyHostToDevice);
	check(status,"Memcpy queries failed\n");


	status = cudaMalloc((void **)&sources_dev, source_nb * dim * sizeof(float));
	check(status,"Malloc sources failed\n");
	status = cudaMemcpy(sources_dev, sources, source_nb * dim * sizeof(float), cudaMemcpyHostToDevice);
	check(status,"Mem sources failed\n");

	status = cudaMalloc((void **)&qreps_dev, qrep_nb * dim * sizeof(float));
	check(status,"Malloc reps failed\n");
	//status = cudaMemcpy(qreps_dev, qreps, qrep_nb * dim * sizeof(float), cudaMemcpyHostToDevice);
	//check(status,"Mem reps failed\n");

	status = cudaMalloc((void **)&sreps_dev, srep_nb * dim * sizeof(float));
	check(status,"Malloc reps failed\n");
	//status = cudaMemcpy(sreps_dev, sreps, srep_nb * dim * sizeof(float), cudaMemcpyHostToDevice);
	//check(status,"Mem reps failed\n");

	int totalTest = 10;
	int *qIndex_dev, *qIndex;
	qIndex = (int *)malloc(totalTest * qrep_nb * sizeof(int));
	cudaMalloc((void **)&qIndex_dev, qrep_nb * totalTest * sizeof(int));
	srand(2015);
	for(int i = 0;i<totalTest;i++)
		for(int j =0 ;j<qrep_nb;j++)
			qIndex[i*qrep_nb +j]=rand()%query_nb;
	cudaMemcpy(qIndex_dev, qIndex, totalTest * qrep_nb * sizeof(int), cudaMemcpyHostToDevice);
	int *totalSum, *totalSum_dev;
	cudaMalloc((void **)&totalSum_dev, totalTest * sizeof(float));
	cudaMemset(totalSum_dev, 0, totalTest*sizeof(float));
	totalSum = (int *)malloc(totalTest*sizeof(float));




	selectReps_cuda<<<(totalTest*qrep_nb*qrep_nb+255)/256,256>>>(queries_dev, query_nb, qreps_dev, qrep_nb, qIndex_dev, totalSum_dev, totalTest, dim);

	cudaDeviceSynchronize();
	print_last_error();


	selectReps_max<<<1,1>>>(queries_dev, query_nb, qreps_dev, qrep_nb, qIndex_dev, totalSum_dev, totalTest, dim);
	selectReps_copy<<<(qrep_nb + 255)/256,256>>>(queries_dev, query_nb, qreps_dev, qrep_nb, qIndex_dev, totalSum_dev, totalTest, dim);



	qIndex = (int *)malloc(totalTest * srep_nb * sizeof(int));
	cudaMalloc((void **)&qIndex_dev, srep_nb * totalTest * sizeof(int));

	srand(2015);
	for(int i = 0;i<totalTest;i++)
		for(int j =0 ;j<srep_nb;j++)
			qIndex[i*srep_nb +j]=rand()%source_nb;
	cudaMemcpy(qIndex_dev, qIndex, totalTest * srep_nb * sizeof(int), cudaMemcpyHostToDevice);


	cudaMemset(totalSum_dev, 0, totalTest*sizeof(float));


	selectReps_cuda<<<(totalTest*srep_nb*srep_nb+255)/256,256>>>(sources_dev, source_nb, sreps_dev, srep_nb, qIndex_dev, totalSum_dev, totalTest, dim);
	selectReps_max<<<1,1>>>(sources_dev, source_nb, sreps_dev, srep_nb, qIndex_dev, totalSum_dev, totalTest, dim);

	selectReps_copy<<<(srep_nb + 255)/256,256>>>(sources_dev, source_nb, sreps_dev, srep_nb, qIndex_dev, totalSum_dev, totalTest, dim);
	cudaDeviceSynchronize();	



	cudaMalloc((void **)&rep2q_static_dev, qrep_nb * sizeof(R2all_static_dev));
	check(status,"Malloc rep2qs_static failed\n");
	cudaMemcpy(rep2q_static_dev, rep2q_static, qrep_nb * sizeof(R2all_static_dev), cudaMemcpyHostToDevice);
	check(status,"Memcpy rep2qs_static failed\n");

	cudaMalloc((void **)&rep2s_static_dev, srep_nb * sizeof(R2all_static_dev));
	check(status,"Malloc rep2qs_static failed\n");
	cudaMemcpy(rep2s_static_dev, rep2s_static, srep_nb * sizeof(R2all_static_dev), cudaMemcpyHostToDevice);
	check(status,"Memcpy rep2qs_static failed\n");





	int block = 256;

	float *queryNorm_dev, *qrepNorm_dev, *sourceNorm_dev, *srepNorm_dev;
	cudaMalloc((void **)&queryNorm_dev,query_nb * sizeof(float));
	cudaMalloc((void **)&sourceNorm_dev,source_nb * sizeof(float));
	cudaMalloc((void **)&qrepNorm_dev, qrep_nb * sizeof(float));
	cudaMalloc((void **)&srepNorm_dev, srep_nb * sizeof(float));


	//cudaDeviceSynchronize();
	struct timespec t3,t4,t35;
	timePoint(t3);
	cublasSgemm('T','N', query_nb, qrep_nb, dim, (float)-2.0, queries_dev,dim, qreps_dev,dim,(float)0.0,query2reps_dev,query_nb);
	cudaDeviceSynchronize();
	timePoint(t35);
	printf("cublasSgemm warm up time %f\n", timeLen(t3,t35));
	timePoint(t1);
	Norm<<<(query_nb + 255)/256,256>>>(queries_dev, queryNorm_dev, query_nb, dim);

	cublasSgemm('T','N', query_nb, qrep_nb, dim, (float)-2.0, queries_dev,dim, qreps_dev,dim,(float)0.0,query2reps_dev,query_nb);

	cudaDeviceSynchronize();
	timePoint(t3);
	Norm<<<(qrep_nb + 255)/256, 256>>>(qreps_dev,qrepNorm_dev,qrep_nb,dim);
	dim3 block2D(16,16,1);
	dim3 grid2D_q((query_nb+15)/16,(qrep_nb+15)/16,1);
	AddAll<<<grid2D_q,block2D>>>(queryNorm_dev,qrepNorm_dev,query2reps_dev,query_nb, qrep_nb);
	//cudaMemcpy(query2reps, query2reps_dev, rep_nb * query_nb * sizeof(float), cudaMemcpyDeviceToHost);



	cudaMalloc((void **)&maxquery_dev, qrep_nb * sizeof(float));
	cudaMemset(maxquery_dev,0,qrep_nb * sizeof(float));


	status = cudaMalloc((void **)&q2rep_dev, query_nb * sizeof(P2R));
	check(status,"Malloc q2rep failed\n");
	findQCluster<<<(query_nb + 255)/256,256>>>(query2reps_dev, q2rep_dev, query_nb, qrep_nb, maxquery_dev,rep2q_static_dev);

	timePoint(t35);
	printf("query rep first part time %f\n",timeLen(t3,t35));

	int  *qrepsID;
	cudaMalloc((void **)&qrepsID, qrep_nb * sizeof(int));
	cudaMemset(qrepsID, 0, qrep_nb * sizeof(int));
	cudaMemcpy(rep2q_static, rep2q_static_dev, qrep_nb * sizeof(R2all_static_dev), cudaMemcpyDeviceToHost);
	check(status,"Memcpy rep2qs_static failed\n");
	for(int i = 0; i <qrep_nb; i++){
		cudaMalloc((void **)&rep2q_dyn_p[i].replist, srep_nb * sizeof(IndexDist));
		cudaMalloc((void **)&rep2q_dyn_p[i].kubound, K * sizeof(float));
		cudaMalloc((void **)&rep2q_dyn_p[i].memberID, rep2q_static[i].npoints * sizeof(int));
	}

	cudaMalloc((void **)&rep2q_dyn_p_dev, qrep_nb * sizeof(R2all_dyn_p));
	cudaMemcpy(rep2q_dyn_p_dev, rep2q_dyn_p, qrep_nb * sizeof(R2all_dyn_p), cudaMemcpyHostToDevice);
	fillQMembers<<<(query_nb + 255)/256,256>>>(q2rep_dev, query_nb, qrepsID, rep2q_dyn_p_dev);


	cudaMalloc((void **)&reorder_members, query_nb * sizeof(int));
	
	reorderMembers<<<(qrep_nb + 255)/256,256>>>(qrep_nb, qrepsID, reorder_members, rep2q_dyn_p_dev);
	//cudaDeviceSynchronize();


	cudaDeviceSynchronize();
	timePoint(t4);
	printf("query rep time  %f\n",timeLen(t3,t4));
	float *source2reps = (float *)malloc(source_nb * srep_nb * sizeof(float));
	float *source2reps_dev;
	cudaMalloc((void **)&source2reps_dev,source_nb * srep_nb * sizeof(float));

	cudaDeviceSynchronize();
	timePoint(t3);
	Norm<<<(source_nb + 255)/256,256>>>(sources_dev, sourceNorm_dev, source_nb, dim);
	cublasSgemm('T','N', source_nb, srep_nb, dim, (float)-2.0, sources_dev,dim, sreps_dev,dim,(float)0.0,source2reps_dev,source_nb);
	cudaDeviceSynchronize();
	timePoint(t35);
	printf("source rep first part time %f\n",timeLen(t3,t35));
	Norm<<<(srep_nb + 255)/256, 256>>>(sreps_dev,srepNorm_dev,srep_nb,dim);
	dim3 grid2D_s((source_nb+15)/16,(srep_nb+15)/16,1);
	AddAll<<<grid2D_s,block2D>>>(sourceNorm_dev,srepNorm_dev,source2reps_dev,source_nb, srep_nb);

	status = cudaMalloc((void **)&s2rep_dev, source_nb * sizeof(P2R));
	check(status,"Malloc s2rep failed\n");
	findTCluster<<<(source_nb + 255)/256,256>>>(source2reps_dev, s2rep_dev, source_nb, srep_nb,rep2s_static_dev);
	int  *srepsID;
	cudaMalloc((void **)&srepsID, srep_nb * sizeof(int));
	cudaMemset(srepsID, 0, srep_nb * sizeof(int));
	cudaMemcpy(rep2s_static, rep2s_static_dev, srep_nb * sizeof(R2all_static_dev), cudaMemcpyDeviceToHost);
	for(int i = 0;i<srep_nb;i++){
		cudaMalloc((void **)&rep2s_dyn_p[i].sortedmembers, rep2s_static[i].npoints * sizeof(R2all_dyn_p));
	}
	cudaMalloc((void **)&rep2s_dyn_p_dev, srep_nb * sizeof(R2all_dyn_p));
	cudaMemcpy(rep2s_dyn_p_dev, rep2s_dyn_p, srep_nb * sizeof(R2all_dyn_p),cudaMemcpyHostToDevice);
	fillTMembers<<<(source_nb + 255)/256,256>>>(s2rep_dev, source_nb, srepsID, rep2s_dyn_p_dev);

	/*
	cudaMemcpy(source2reps, source2reps_dev, srep_nb * source_nb * sizeof(float), cudaMemcpyDeviceToHost);
	for(int i = 0; i < source_nb; i++){
		float distance = FLT_MAX;
		int repIndex = -1;
		for(int j = 0; j < srep_nb; j++){
			float len = source2reps[j * source_nb + i];//Edistance(getPoint(sources,i), getPoint(reps,j));
			if(distance > len){
				distance = len;
				repIndex = j;
			}
		}
		s2rep[i].repIndex = repIndex;
		s2rep[i].dist2rep = distance;
		*/
/*
		rep2qs_dyn_v[repIndex].VsortedIndex.push_back(i);
		rep2qs_dyn_v[repIndex].VsortedDist.push_back(distance);
		*/
/*	
		IndexDist temp = {i, distance};
		rep2s_dyn_v[repIndex].Vsortedmembers.push_back(temp);
	
	}
	*/
	timePoint(t3);
	
	//cudaStream_t *streamID = (cudaStream_t *)malloc(srep_nb * sizeof(cudaStream_t));
	#pragma omp parallel for
	for(int i = 0; i < srep_nb; i++){
		//cudaStreamCreate(&streamID[i]);
		if(rep2s_static[i].npoints>0){

			vector<IndexDist> temp;
			temp.resize(rep2s_static[i].npoints);
			cudaMemcpy(&temp[0],rep2s_dyn_p[i].sortedmembers, rep2s_static[i].npoints * sizeof(IndexDist), cudaMemcpyDeviceToHost);
			sort(temp.begin(),temp.end(),sort_inc());
			//rep2s_static[i].maxdist = temp[rep2s_static[i].npoints-1].dist;
			//rep2s_static[i].mindist = temp[0].dist;
			cudaMemcpy(rep2s_dyn_p[i].sortedmembers, &temp[0], rep2s_static[i].npoints * sizeof(IndexDist), cudaMemcpyHostToDevice);
#if debug
			cout<<"max "<<rep2qs_static[i].maxsource<<" min: "<<rep2qs_static[i].minsource<<" Qpoints:"<<rep2qs_static[i].noqueries<<" Spoints:"<<rep2qs_static[i].nosources<<endl;
	
#endif
		}
	}
	
	timePoint(t4);
	cudaFree(query2reps_dev);
	cudaMalloc((void **)&query2reps_dev, query_nb * srep_nb * sizeof(float));
	dim3 grid2D_qsrep((query_nb+15)/16,(srep_nb+15)/16,1);
	    cublasSgemm('T','N', query_nb, srep_nb, dim, (float)-2.0, queries_dev,dim, sreps_dev,dim,(float)0.0,query2reps_dev,query_nb);
		AddAll<<<grid2D_qsrep,block2D>>>(queryNorm_dev,srepNorm_dev,query2reps_dev,query_nb, srep_nb);
	//cudaDeviceSynchronize();

	printf("source rep time %f\n",timeLen(t3,t4));

}

void AllocateAndCopyH2D(float * &queries_dev, float * &sources_dev, float * &qreps_dev, float * &sreps_dev, float *maxquery_dev,
	P2R * &q2rep_dev, P2R * &s2rep_dev, R2all_static_dev * &rep2q_static_dev, R2all_static_dev * &rep2s_static_dev, R2all_dyn_p * &rep2q_dyn_p_dev, R2all_dyn_p * &rep2s_dyn_p_dev, float * &query2reps_dev,
	P2R * &q2rep,     P2R * &s2rep,     R2all_static * &rep2q_static,         R2all_static * &rep2s_static,     R2all_dyn_v * &rep2q_dyn_v,		R2all_dyn_v * &rep2s_dyn_v,     float * &query2reps,
	R2all_dyn_p * &rep2q_dyn_p, R2all_dyn_p * &rep2s_dyn_p){

	cudaError_t status;
/*
	status = cudaMemcpy(q2rep_dev, q2rep, query_nb * sizeof(P2R), cudaMemcpyHostToDevice);
	check(status,"Memcpy reps failed\n");
*/

	//status = cudaMemcpy(s2rep_dev, s2rep, source_nb * sizeof(P2R), cudaMemcpyHostToDevice);
	//check(status,"Memcpy s2rep failed\n");

	cudaMemcpy(rep2q_static_dev, rep2q_static, qrep_nb * sizeof(R2all_static_dev), cudaMemcpyHostToDevice);
	check(status,"Memcpy rep2qs_static failed\n");

	cudaMemcpy(rep2s_static_dev, rep2s_static, srep_nb * sizeof(R2all_static_dev), cudaMemcpyHostToDevice);
	check(status,"Memcpy rep2qs_static failed\n");




	printf("sizeof static static_dev %d %d\n", sizeof(R2all_static), sizeof(R2all_static_dev));


/*

	for(int i = 0; i < srep_nb; i++){
		int nosources_local = rep2s_static[i].npoints;

		if(nosources_local > 0){
			status = cudaMalloc((void **)&rep2s_dyn_p[i].sortedmembers, nosources_local * sizeof(IndexDist));
			check(status,"Malloc rep2qs_dyn source failed\n");
			status = cudaMemcpy(rep2s_dyn_p[i].sortedmembers, &rep2s_dyn_v[i].Vsortedmembers[0], nosources_local * sizeof(IndexDist), cudaMemcpyHostToDevice);
			check(status,"Memcpy rep2qs_dyn source failed\n");

		}
	}

	cudaMalloc((void **)&rep2s_dyn_p_dev, srep_nb * sizeof(R2all_dyn_p));
	cudaMemcpy(rep2s_dyn_p_dev, rep2s_dyn_p, srep_nb * sizeof(R2all_dyn_p), cudaMemcpyHostToDevice);
	*/

}

__global__ void RepsUpperBound(float *queries_dev, float *sources_dev, float *qreps_dev, float *sreps_dev, float *query2reps_dev, float *maxquery_dev,
			P2R *q2rep_dev, P2R *s2rep_dev, R2all_static_dev *rep2q_static_dev, R2all_dyn_p *rep2q_dyn_p_dev,  R2all_static_dev *rep2s_static_dev, R2all_dyn_p *rep2s_dyn_p_dev,
			int query_nb, int source_nb, int qrep_nb, int srep_nb, int dim, int K){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < qrep_nb){
		//if(fabs(maxquery_dev[tid]-rep2qs_static_dev[tid].maxquery)>0.01)
		//	printf("tid %d %.10f %.10f\n",tid, maxquery_dev[tid],rep2qs_static_dev[tid].maxquery);
		int UBoundCount = 0;
		for(int i = 0; i < srep_nb; i++){
			float rep2rep = Edistance_128(qreps_dev + tid * dim, sreps_dev + i * dim, dim);
			int count = 0;
			while(count < K && count < rep2s_static_dev[i].npoints){


				//float g2pUBound = rep2qs_static_dev[tid].maxquery + rep2rep + rep2qs_dyn_p_dev[i].sortedsources[count].dist;
				float g2pUBound = maxquery_dev[tid] + rep2rep + rep2s_dyn_p_dev[i].sortedmembers[count].dist;

				if(UBoundCount < K){
					rep2q_dyn_p_dev[tid].kubound[UBoundCount] = g2pUBound;
					if(rep2q_static_dev[tid].kuboundMax < g2pUBound)
						rep2q_static_dev[tid].kuboundMax = g2pUBound;

					UBoundCount++;
				}
				else{
					if(rep2q_static_dev[tid].kuboundMax > g2pUBound){
						float max_local = 0.0f;
						for(int j = 0; j < K; j++){
							if(rep2q_dyn_p_dev[tid].kubound[j]==rep2q_static_dev[tid].kuboundMax){
								rep2q_dyn_p_dev[tid].kubound[j] = g2pUBound;
							}
							if(max_local < rep2q_dyn_p_dev[tid].kubound[j]){
								max_local = rep2q_dyn_p_dev[tid].kubound[j];
							}
						}
						rep2q_static_dev[tid].kuboundMax = max_local;
					}
				}
				count++;
			}
		}
#if debug
		printf("i = %d, %.10f\n",tid,rep2qs_static_dev[tid].kuboundMax);
#endif
	}
}

__global__ void FilterReps
			(float *queries_dev, float *sources_dev, float *qreps_dev, float *sreps_dev, float *query2reps_dev, float *maxquery_dev,
			P2R *q2rep_dev, P2R *s2rep_dev, R2all_static_dev *rep2q_static_dev, R2all_dyn_p *rep2q_dyn_p_dev,  R2all_static_dev *rep2s_static_dev, R2all_dyn_p *rep2s_dyn_p_dev,
			int query_nb, int source_nb, int qrep_nb, int srep_nb, int dim, int K){
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;   //calculate reps[tidy].replist;
	if(tidx < srep_nb && tidy < qrep_nb){
		float distance = Edistance(qreps_dev + tidy * dim, sreps_dev + tidx * dim, dim);
		//if(distance - rep2qs_static_dev[tidy].maxquery - rep2qs_static_dev[tidx].maxsource < rep2qs_static_dev[tidy].kuboundMax){
		if(distance - maxquery_dev[tidy] - rep2s_static_dev[tidx].maxdist < rep2q_static_dev[tidy].kuboundMax){

			int rep_id = atomicAdd(&rep2q_static_dev[tidy].noreps,1);
			rep2q_dyn_p_dev[tidy].replist[rep_id].index = tidx;
			rep2q_dyn_p_dev[tidy].replist[rep_id].dist = distance;
#if debug
			printf("tidy = %d tidx = %d distance = %.10f\n", tidy, tidx, distance);
#endif
		}
	}
}


__global__ void NearReps(float *queries_dev, float *sources_dev, float *reps_dev, float *query2reps_dev,
			P2R *q2rep_dev, P2R *s2rep_dev, R2all_static_dev *rep2qs_static_dev, R2all_dyn_p *rep2qs_dyn_p_dev,
			int query_nb, int source_nb, int rep_nb, int dim, int K){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < rep_nb){
		IndexDist *temp = rep2qs_dyn_p_dev[tid].replist;
		float max_local = 0.0f;
		int index = -1;
		for(int i = 0; i < rep2qs_static_dev[tid].noreps; i++){
			if(max_local < temp[i].dist){
				max_local = temp[i].dist;
				index = i;
			}
		}
		IndexDist tmp;
		tmp = temp[index];
		temp[index] = temp[0];
		temp[0] = tmp;
	}
}
__global__ void SortReps(float *queries_dev, float *sources_dev, float *reps_dev, float *query2reps_dev,
			P2R *q2rep_dev, P2R *s2rep_dev, R2all_static_dev *rep2qs_static_dev, R2all_dyn_p *rep2qs_dyn_p_dev,
			int query_nb, int source_nb, int rep_nb, int dim, int K){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < rep_nb){
		IndexDist *temp = rep2qs_dyn_p_dev[tid].replist;
		for(int i = 0; i < rep2qs_static_dev[tid].noreps; i++)
			for(int j = i; j < rep2qs_static_dev[tid].noreps; j++){
				if(temp[i].dist > temp[j].dist){
					IndexDist tmp = temp[j];
					temp[j] = temp[i];
					temp[i] = tmp;
				}
			}
	}
				
}
__device__ int Total = 0;
__global__ void printTotal(){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid == 0)
		printf("Total %d\n",Total);
}
__global__ void KNNQuery_base3
			(float *queries_dev, float *sources_dev, float *qreps_dev, float *sreps_dev, float *query2reps_dev, float *maxquery_dev,
			P2R *q2rep_dev, P2R *s2rep_dev, R2all_static_dev *rep2q_static_dev, R2all_dyn_p *rep2q_dyn_p_dev,  R2all_static_dev *rep2s_static_dev, R2all_dyn_p *rep2s_dyn_p_dev,
			int query_nb, int source_nb, int qrep_nb, int srep_nb, int dim, int K, IndexDist * knearest, int * reorder_members){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < query_nb){
		tid = reorder_members[tid];
		int repIndex = q2rep_dev[tid].repIndex;
		float theta = rep2q_static_dev[repIndex].kuboundMax;
		int Kcount = 0;
		int count = 0;

		for(int i = 0; i < rep2q_static_dev[repIndex].noreps; i++){
			int minlb_rid = rep2q_dyn_p_dev[repIndex].replist[i].index;
			float query2rep = 0.0f;
			//if(repIndex != minlb_rid){
				query2rep = query2reps_dev[tid + minlb_rid*query_nb]; 
							//Edistance_128(queries_dev + tid * dim, sreps_dev + minlb_rid * dim, dim);
				atomicAdd(&Total,1);
			//}
			//else
			//	query2rep = q2rep_dev[tid].dist2rep;

			for(int j = rep2s_static_dev[minlb_rid].npoints - 1; j >= 0; j--){


				IndexDist sourcej = rep2s_dyn_p_dev[minlb_rid].sortedmembers[j];
#if debug
				if(tid == 0)
					printf("j %d %.10f\n",sourcej.index,sourcej.dist);
#endif


				float p2plbound = query2rep - sourcej.dist;
				if(p2plbound > theta)
					break;
				else if(p2plbound < theta*(-1.0f))
					continue;
				else if(p2plbound <= theta && p2plbound >= theta*(-1.0f)){
					float query2source = Edistance_128(queries_dev + tid * dim, sources_dev + sourcej.index * dim, dim);
					count++;
					//atomicAdd(&Total, 1);

#if debug
				if(tid == 0){

					printf("query2source %.10f %.10f %.10f\n", query2source, p2plbound, theta);
				}
#endif

					int insert = -1;
					float max_local = 0.0f;
					for( int kk = 0; kk < Kcount; kk++){
						if(query2source < knearest[tid * K + kk].dist){
							insert = kk;
							break;
						}
					}
					if(Kcount < K){
						if(insert == -1){
							knearest[tid * K + Kcount] = {sourcej.index, query2source};
						}
						else{
							for(int move = Kcount - 1; move >= insert; move--){
								knearest[tid * K + (move + 1)] = knearest[tid * K + move];
							}
							knearest[tid * K + insert] = {sourcej.index, query2source};
						}
						Kcount++;
					}
					else{  //Kcount = K
						if(insert == -1)
							continue;
						else{
							for(int move = K - 2; move >= insert; move--){
								knearest[tid * K + (move + 1)] = knearest[tid * K + move];
							}

							knearest[tid * K + insert] = {sourcej.index, query2source};
							theta = knearest[(K - 1) + tid * K].dist;
						}

					}
					
				}
			}
		}
		//memcpy(&knearest1[tid * K], knearest, 20 * sizeof(IndexDist));
		/*
		if(tid == 100)
			for(int i = 0; i < K; i++)
				printf("tid i Index Dist %d %d %d %.10f\n",tid, i, knearest[tid * K + i].index, knearest[tid * K +i].dist);
		*/
		
	}
}
__global__ void KNNQuery_base2
			(float *queries_dev, float *sources_dev, float *qreps_dev, float *sreps_dev, float *query2reps_dev, float *maxquery_dev,
			P2R *q2rep_dev, P2R *s2rep_dev, R2all_static_dev *rep2q_static_dev, R2all_dyn_p *rep2q_dyn_p_dev,  R2all_static_dev *rep2s_static_dev, R2all_dyn_p *rep2s_dyn_p_dev,
			int query_nb, int source_nb, int qrep_nb, int srep_nb, int dim, int K, IndexDist * knearest, int * reorder_members){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < query_nb){
		tid = reorder_members[tid];
		int repIndex = q2rep_dev[tid].repIndex;
		float theta = rep2q_static_dev[repIndex].kuboundMax;
		int Kcount = 0;
		int count = 0;

		for(int i = 0; i < rep2q_static_dev[repIndex].noreps; i++){
			int minlb_rid = rep2q_dyn_p_dev[repIndex].replist[i].index;
			float query2rep = 0.0f;
			//if(repIndex != minlb_rid){
				query2rep = query2reps_dev[tid + minlb_rid*query_nb]; 
							//Edistance_128(queries_dev + tid * dim, sreps_dev + minlb_rid * dim, dim);
				atomicAdd(&Total,1);
			//}
			//else
			//	query2rep = q2rep_dev[tid].dist2rep;

			for(int j = rep2s_static_dev[minlb_rid].npoints - 1; j >= 0; j--){


				IndexDist sourcej = rep2s_dyn_p_dev[minlb_rid].sortedmembers[j];
#if debug
				if(tid == 0)
					printf("j %d %.10f\n",sourcej.index,sourcej.dist);
#endif


				float p2plbound = query2rep - sourcej.dist;
				if(p2plbound > theta)
					break;
				else if(p2plbound < theta*(-1.0f))
					continue;
				else if(p2plbound <= theta && p2plbound >= theta*(-1.0f)){
					float query2source = Edistance_128(queries_dev + tid * dim, sources_dev + sourcej.index * dim, dim);
					count++;
					//atomicAdd(&Total, 1);

#if debug
				if(tid == 0){

					printf("query2source %.10f %.10f %.10f\n", query2source, p2plbound, theta);
				}
#endif

					int insert = -1;
					float max_local = 0.0f;
					for( int kk = 0; kk < Kcount; kk++){
						if(query2source < knearest[tid + kk * query_nb].dist){
							insert = kk;
							break;
						}
					}
					if(Kcount < K){
						if(insert == -1){
							knearest[tid + Kcount * query_nb] = {sourcej.index, query2source};
						}
						else{
							for(int move = Kcount - 1; move >= insert; move--){
								knearest[tid + (move + 1) * query_nb] = knearest[tid + move * query_nb];
							}
							knearest[tid + insert * query_nb] = {sourcej.index, query2source};
						}
						Kcount++;
					}
					else{  //Kcount = K
						if(insert == -1)
							continue;
						else{
							for(int move = K - 2; move >= insert; move--){
								knearest[tid + (move + 1)*query_nb] = knearest[tid + move*query_nb];
							}

							knearest[tid + insert * query_nb] = {sourcej.index, query2source};
							theta = knearest[(K - 1)*query_nb + tid].dist;
						}

					}
				}
			}
		}
		//memcpy(&knearest1[tid * K], knearest, 20 * sizeof(IndexDist));
		/*
		if(tid == 100)
			for(int i = 0; i < K; i++)
				printf("tid i Index Dist %d %d %d %.10f\n",tid, i, knearest[tid * K + i].index, knearest[tid * K +i].dist);
		*/
		
	}
}
__global__ void KNNQuery_base
			(float *queries_dev, float *sources_dev, float *qreps_dev, float *sreps_dev, float *query2reps_dev, float *maxquery_dev,
			P2R *q2rep_dev, P2R *s2rep_dev, R2all_static_dev *rep2q_static_dev, R2all_dyn_p *rep2q_dyn_p_dev,  R2all_static_dev *rep2s_static_dev, R2all_dyn_p *rep2s_dyn_p_dev,
			int query_nb, int source_nb, int qrep_nb, int srep_nb, int dim, int K, IndexDist * knearest1, int * reorder_members){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < query_nb){
		tid = reorder_members[tid];
		int repIndex = q2rep_dev[tid].repIndex;
		float theta = rep2q_static_dev[repIndex].kuboundMax;
		int Kcount = 0;
		int count = 0;

		IndexDist knearest[1000];
		for(int i = 0; i < rep2q_static_dev[repIndex].noreps; i++){
			int minlb_rid = rep2q_dyn_p_dev[repIndex].replist[i].index;
			float query2rep = 0.0f;
			//if(repIndex != minlb_rid){
				query2rep = query2reps_dev[tid + minlb_rid*query_nb]; 
							//Edistance_128(queries_dev + tid * dim, sreps_dev + minlb_rid * dim, dim);
		 	//	atomicAdd(&Total,1);
			//}
			//else
			//	query2rep = q2rep_dev[tid].dist2rep;

			for(int j = rep2s_static_dev[minlb_rid].npoints - 1; j >= 0; j--){


				IndexDist sourcej = rep2s_dyn_p_dev[minlb_rid].sortedmembers[j];
#if debug
				if(tid == 0)
					printf("j %d %.10f\n",sourcej.index,sourcej.dist);
#endif


				float p2plbound = query2rep - sourcej.dist;
				if(p2plbound > theta)
					break;
				else if(p2plbound < theta*(-1.0f))
					continue;
				else if(p2plbound <= theta && p2plbound >= theta*(-1.0f)){
					float query2source = Edistance_128(queries_dev + tid * dim, sources_dev + sourcej.index * dim, dim);
					count++;
					atomicAdd(&Total, 1);

#if debug
				if(tid == 0){

					printf("query2source %.10f %.10f %.10f\n", query2source, p2plbound, theta);
				}
#endif

					int insert = -1;
					float max_local = 0.0f;
					for( int kk = 0; kk < Kcount; kk++){
						if(query2source < knearest[kk].dist){
							insert = kk;
							break;
						}
					}
					if(Kcount < K){
						if(insert == -1){
							knearest[Kcount] = {sourcej.index, query2source};
						}
						else{
							for(int move = Kcount - 1; move >= insert; move--){
								knearest[move + 1] = knearest[move];
							}
							knearest[insert] = {sourcej.index, query2source};
						}
						Kcount++;
					}
					else{  //Kcount = K
						if(insert == -1)
							continue;
						else{
							for(int move = K - 2; move >= insert; move--){
								knearest[move + 1] = knearest[move];
							}

							knearest[insert] = {sourcej.index, query2source};
							theta = knearest[K - 1].dist;
						}

					}
				}
			}
		}
		memcpy(&knearest1[tid * K], knearest, K * sizeof(IndexDist));
		
		/*
		if(tid == 100)
			for(int i = 0; i < K; i++)
				printf("tid i Index Dist %d %d %d %.10f\n",tid, i, knearest[tid * K + i].index, knearest[tid * K +i].dist);
		*/
		
	}
}
__global__ void KNNQuery_theta
			(float *queries_dev, float *sources_dev, float *qreps_dev, float *sreps_dev, float *query2reps_dev, float *maxquery_dev,
			P2R *q2rep_dev, P2R *s2rep_dev, R2all_static_dev *rep2q_static_dev, R2all_dyn_p *rep2q_dyn_p_dev,  R2all_static_dev *rep2s_static_dev, R2all_dyn_p *rep2s_dyn_p_dev,
			int query_nb, int source_nb, int qrep_nb, int srep_nb, int dim, int K, IndexDist * knearest, float * thetas){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < query_nb){
		int repIndex = q2rep_dev[tid].repIndex;
		thetas[tid] = rep2q_static_dev[repIndex].kuboundMax;
	}
}
__global__ void KNNQuery
			(float *queries_dev, float *sources_dev, float *qreps_dev, float *sreps_dev, float *query2reps_dev, float *maxquery_dev,
			P2R *q2rep_dev, P2R *s2rep_dev, R2all_static_dev *rep2q_static_dev, R2all_dyn_p *rep2q_dyn_p_dev,  R2all_static_dev *rep2s_static_dev, R2all_dyn_p *rep2s_dyn_p_dev,
			int query_nb, int source_nb, int qrep_nb, int srep_nb, int dim, int K, IndexDist * knearest, float * thetas, int tpq, int *reorder_members ){
	int ttid = threadIdx.x + blockIdx.x * blockDim.x;
	int tp = ttid % tpq;
	int tid = ttid / tpq;
	if(tid < query_nb){
		tid = reorder_members[tid];
		ttid = tid * tpq + tp;
		int repIndex = q2rep_dev[tid].repIndex;
		//float theta = rep2q_static_dev[repIndex].kuboundMax;
		int Kcount = 0;
		int count = 0;

		for(int i = 0; i < rep2q_static_dev[repIndex].noreps; i++){
			int minlb_rid = rep2q_dyn_p_dev[repIndex].replist[i].index;
			float query2rep = 0.0f;
			//if(repIndex != minlb_rid){
				query2rep = //query2reps_dev[tid + minlb_rid*query_nb]; 
							Edistance_128(queries_dev + tid * dim, sreps_dev + minlb_rid * dim, dim);
				//atomicAdd(&Total,1);

			for(int j = rep2s_static_dev[minlb_rid].npoints - 1 - tp; j >= 0; j-=tpq){
				IndexDist sourcej = rep2s_dyn_p_dev[minlb_rid].sortedmembers[j];
#if debug
				if(tid == 0)
					printf("j %d %.10f\n",sourcej.index,sourcej.dist);
#endif
				float p2plbound = query2rep - sourcej.dist;
				if(p2plbound > *(volatile float *)&thetas[tid])
					break;
				else if(p2plbound < *(volatile float *)&thetas[tid] *(-1.0f))
					continue;
				else if(p2plbound <= *(volatile float *)&thetas[tid] && p2plbound >= *(volatile float *)&thetas[tid]*(-1.0f)){
					float query2source = Edistance_128(queries_dev + tid * dim, sources_dev + sourcej.index * dim, dim);
					count++;
					atomicAdd(&Total, 1);

#if debug
				if(tid == 0){

					printf("query2source %.10f %.10f %.10f\n", query2source, p2plbound, theta);
				}
#endif

					int insert = -1;
					float max_local = 0.0f;
					for( int kk = 0; kk < Kcount; kk++){
						if(query2source < knearest[ttid * K + kk].dist){
							insert = kk;
							break;
						}
					}
					if(Kcount < K){
						if(insert == -1){
							knearest[ttid * K + Kcount] = {sourcej.index, query2source};
						}
						else{
							for(int move = Kcount - 1; move >= insert; move--){
								knearest[ttid * K + move + 1] = knearest[ttid * K + move];
							}
							knearest[ttid * K + insert] = {sourcej.index, query2source};
						}
						Kcount++;
					}
					else{  //Kcount = K
						if(insert == -1)
							continue;
						else{
							for(int move = K - 2; move >= insert; move--){
								knearest[ttid * K + move + 1] = knearest[ttid * K + move];
							}

							knearest[ttid * K + insert] = {sourcej.index, query2source};
							atomicMin_float(&thetas[tid], knearest[ttid * K + K - 1].dist);
						}

					}
				}
			}
		}
		/*
		if(tid == 100)
			for(int i = 0; i < K; i++)
				printf("tid i Index Dist %d %d %d %.10f\n",tid, i, knearest[tid * K + i].index, knearest[tid * K +i].dist);
				*/
		
	}
}

__global__ void final(int k, IndexDist * knearest, int tpq, int query_nb, IndexDist *final_knearest, int *tag_base){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int *tag = tid * tpq + tag_base;
	if(tid < query_nb){
	for(int i = 0; i < k; i++){
		float min = knearest[tid * tpq * k + tag[0]].dist;
		int index = 0;
		for(int j = 1; j < tpq ; j++){
			float value = knearest[(tid * tpq + j)* k + tag[j]].dist;
			if(min > value){
				min = value;
				index = j;
			}
		}
		//if(tid ==100) printf("final i index tag %d %d %d %f\n",i, index, tag[14],knearest[(tid * tpq + 14)* k + tag[14]].dist);
		final_knearest[tid * k + i] = knearest[(tid * tpq + index)* k + tag[index]];
		tag[index]++;
	}
	}
}
void *work(void *para){
	cudaFree(0);
}
int main(int argc, char *argv[]){
	pthread_t thread2;
	timePoint(t1);
	int rc = pthread_create(&thread2,NULL, work, NULL);
	//cudaFree(0);


	if(argc<9){
		cout<<"usage: ./exe query_nb source_nb dimension rep_nb k input1 input2\n";
		exit(1);
	}
	query_nb = atoi(argv[1]);
	source_nb = atoi(argv[2]);
	dim = atoi(argv[3]);
	qrep_nb = atoi(argv[4]);
	srep_nb = atoi(argv[5]);
	K = atoi(argv[6]);
	char *query_data = argv[7];
	char *source_data = argv[8];

	sources = (float *)malloc(source_nb * dim * sizeof(float));
	queries = (float *)malloc(query_nb * dim * sizeof(float));

	//Setup for source and query points.
	pointSetup(query_data, source_data);

	qreps = (float *)malloc(qrep_nb * dim * sizeof(float));
	sreps = (float *)malloc(srep_nb * dim * sizeof(float));
	P2R *q2rep = (P2R *)malloc(query_nb * sizeof(P2R));
	P2R *s2rep = (P2R *)malloc(source_nb * sizeof(P2R));
	R2all_static *rep2q_static = (R2all_static*)malloc(qrep_nb * sizeof(R2all_static));
	R2all_static *rep2s_static = (R2all_static*)malloc(srep_nb * sizeof(R2all_static));
	R2all_dyn_v *rep2q_dyn_v = (R2all_dyn_v *)malloc(qrep_nb * sizeof(R2all_dyn_v));
	R2all_dyn_v *rep2s_dyn_v = (R2all_dyn_v *)malloc(srep_nb * sizeof(R2all_dyn_v));

	float *query2reps = (float *)malloc(query_nb * qrep_nb * sizeof(float));


	float *queries_dev, *sources_dev, *qreps_dev, *sreps_dev;
	P2R *q2rep_dev, *s2rep_dev;
	R2all_static_dev *rep2q_static_dev;
	R2all_dyn_p *rep2q_dyn_p_dev;
	R2all_static_dev *rep2s_static_dev;
	R2all_dyn_p *rep2s_dyn_p_dev;
	float *query2reps_dev;
	float *maxquery_dev;

	int *reorder_members;

	R2all_dyn_p *rep2q_dyn_p = (R2all_dyn_p *)malloc(qrep_nb * sizeof(R2all_dyn_p));
	R2all_dyn_p *rep2s_dyn_p = (R2all_dyn_p *)malloc(srep_nb * sizeof(R2all_dyn_p));
	//Select reps
	timePoint(t1);
	//selectReps(queries, query_nb, qreps, qrep_nb);
	//selectReps(sources, source_nb, sreps, srep_nb);
	//cluster queries and sources to reps
	 cudaMalloc((void **)&query2reps_dev, qrep_nb * query_nb * sizeof(float));
	//timePoint(t1);
	timePoint(t2);
	printf("cudaFree time %f\n",timeLen(t1,t2));
	clusterReps(queries_dev, sources_dev, qreps_dev, sreps_dev, maxquery_dev,
				q2rep_dev, s2rep_dev, rep2q_static_dev, rep2s_static_dev, rep2q_dyn_p_dev, rep2s_dyn_p_dev, query2reps_dev,
				q2rep,     s2rep,     rep2q_static,     rep2s_static,     rep2q_dyn_v,     rep2s_dyn_v,     query2reps,		rep2q_dyn_p, rep2s_dyn_p, reorder_members);
	//tranfer data structures to GPU.


	AllocateAndCopyH2D(queries_dev, sources_dev, qreps_dev, sreps_dev, maxquery_dev,
				q2rep_dev, s2rep_dev, rep2q_static_dev, rep2s_static_dev, rep2q_dyn_p_dev, rep2s_dyn_p_dev, query2reps_dev,
				q2rep,     s2rep,     rep2q_static,     rep2s_static,     rep2q_dyn_v,     rep2s_dyn_v,     query2reps,		rep2q_dyn_p, rep2s_dyn_p);
	timePoint(t2);
	printf("prepo time %f\n",timeLen(t1,t2));
    if(cudaGetLastError()!=cudaSuccess) cout <<"error 16"<<endl;

	//Kernel 1: upperbound for each rep
	//timePoint(t1);
	RepsUpperBound<<<(qrep_nb+255)/256, 256>>>
											(queries_dev, sources_dev, qreps_dev, sreps_dev, query2reps_dev, maxquery_dev,\
											q2rep_dev, s2rep_dev, rep2q_static_dev, rep2q_dyn_p_dev, rep2s_static_dev, rep2s_dyn_p_dev, \
											query_nb, source_nb, qrep_nb, srep_nb, dim, K);

    if(cudaGetLastError()!=cudaSuccess) cout <<"Kernel RepsUpperBound failed"<<endl;

	//Kernel 2: filter reps	based on upperbound and lowerbound;
	dim3 block(16,16,1);
	dim3 grid((srep_nb+block.x-1)/block.x, (qrep_nb+block.y-1)/block.y,1);
	FilterReps<<<grid, block>>>
								(queries_dev, sources_dev, qreps_dev, sreps_dev, query2reps_dev, maxquery_dev,\
											q2rep_dev, s2rep_dev, rep2q_static_dev, rep2q_dyn_p_dev, rep2s_static_dev, rep2s_dyn_p_dev, \
											query_nb, source_nb, qrep_nb, srep_nb, dim, K);
						
					


	struct timespec sort_start, sort_end;
	timePoint(sort_start);	
	cudaMemcpy(rep2q_static, rep2q_static_dev, qrep_nb * sizeof(R2all_static_dev), cudaMemcpyDeviceToHost);
	
#pragma omp parallel for
	for(int i = 0; i < qrep_nb; i++){
		//printf("replist len %d\n",rep2q_static[i].noreps);
		//IndexDist *tmp = (IndexDist *)malloc(rep2qs_static[i].noreps*sizeof(IndexDist));
		vector<IndexDist> temp;
		temp.resize( rep2q_static[i].noreps);
		cudaMemcpy(&temp[0], rep2q_dyn_p[i].replist, rep2q_static[i].noreps * sizeof(IndexDist), cudaMemcpyDeviceToHost);
		sort(temp.begin(),temp.end(),sort_inc());

		cudaMemcpy(rep2q_dyn_p[i].replist, &temp[0], rep2q_static[i].noreps * sizeof(IndexDist), cudaMemcpyHostToDevice);

	}
	
	timePoint(sort_end);
	printf("sort query replist time %f\n",timeLen(sort_start,sort_end));


	//SortReps<<<(rep_nb + 127) / 128, 128>>>(queries_dev, sources_dev, reps_dev, query2reps_dev,\
											q2rep_dev, s2rep_dev, rep2qs_static_dev, rep2qs_dyn_p_dev, \
											query_nb, source_nb, rep_nb, dim, K);

	//Kernel 3: knn for each point
	IndexDist * knearest, *final_knearest;
	int tpq = (2048*13)/query_nb;
	IndexDist * knearest_h = (IndexDist *)malloc(query_nb * K * sizeof(IndexDist));
	cudaMalloc((void **)&knearest, query_nb * (tpq+1) * K * sizeof(IndexDist));
	int avg_query_nb = int(query_nb / qrep_nb);
	if(tpq>1){
		float *theta;
		cudaMalloc((void **)&theta, query_nb * sizeof(float));
		KNNQuery_theta<<<(query_nb+255)/256, 256>>>
								(queries_dev, sources_dev, qreps_dev, sreps_dev, query2reps_dev, maxquery_dev,\
											q2rep_dev, s2rep_dev, rep2q_static_dev, rep2q_dyn_p_dev, rep2s_static_dev, rep2s_dyn_p_dev, \
											query_nb, source_nb, qrep_nb, srep_nb, dim, K, knearest, theta);
	//cudaMemset(theta, 0, query_nb * sizeof(float));
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

		KNNQuery<<<(tpq*query_nb+255)/256, 256>>>
								(queries_dev, sources_dev, qreps_dev, sreps_dev, query2reps_dev, maxquery_dev,\
											q2rep_dev, s2rep_dev, rep2q_static_dev, rep2q_dyn_p_dev, rep2s_static_dev, rep2s_dyn_p_dev, \
											query_nb, source_nb, qrep_nb, srep_nb, dim, K, knearest, theta, tpq, reorder_members);
		final_knearest = knearest + query_nb * tpq * K;

		int * tag_base;
		cudaMalloc((void **)&tag_base, tpq * query_nb * sizeof(int));
		cudaMemset(tag_base, 0, tpq * query_nb * sizeof(int));
		final<<<(query_nb+255)/256,256>>>(K, knearest,tpq, query_nb, final_knearest, tag_base);
	}
	else{
		KNNQuery_base<<<(query_nb +255)/256, 256>>>
								(queries_dev, sources_dev, qreps_dev, sreps_dev, query2reps_dev, maxquery_dev,\
											q2rep_dev, s2rep_dev, rep2q_static_dev, rep2q_dyn_p_dev, rep2s_static_dev, rep2s_dyn_p_dev, \
											query_nb, source_nb, qrep_nb, srep_nb, dim, K, knearest, reorder_members);
	}
	cudaDeviceSynchronize();
	timePoint(t2);
	printf("total time %f\n",timeLen(t1,t2));
	printTotal<<<1,1>>>();
	if(tpq>1)
		cudaMemcpy(knearest_h, final_knearest, query_nb * K * sizeof(IndexDist),cudaMemcpyDeviceToHost);
	else
		cudaMemcpy(knearest_h, knearest, query_nb * K * sizeof(IndexDist),cudaMemcpyDeviceToHost);
	
	int i = 100;
		for(int j=0;j<K;j++)
			printf("i,k %d %d  %d %f\n",i,j, knearest_h[i*K+j].index,knearest_h[i*K+j].dist);
	/*
	for(int i =0 ;i < 1000;i++)
		for(int j=0;j<K;j++)
			printf("i,k %d %d  %d %f\n",i,j, knearest_h[i*K+j].index,knearest_h[i*K+j].dist);*/
	cudaDeviceSynchronize();





	//R2



	
	free(queries);
	free(sources);
	free(qreps);
	free(sreps);
	free(q2rep);
	free(s2rep);
	free(rep2q_static);
	free(rep2q_dyn_v);
	free(rep2s_static);
	free(rep2s_dyn_v);
	return 0;
}
