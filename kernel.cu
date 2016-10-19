#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdio.h>

#define LENGTH 10
#define TIME	5

#define STEP_X 0.05
#define STEP_T 0.001

#ifdef WIN32
#define GNUPLOT_NAME "pgnuplot -persist"
#else
#define GNUPLOT_NAME "gnuplot -persist"
#endif

static double  *hostData = nullptr, *hostBuffer = nullptr;
static double  *devData  = nullptr, *devBuffer  = nullptr;

static void _free() {
	if (::hostData != nullptr)
		std::free((void *)::hostData);

	if (::hostBuffer != nullptr)
		std::free((void *)::hostBuffer);

	if (::devData != nullptr)
		cudaFree((void *)::devData);

	if (::devBuffer != nullptr)
		cudaFree((void *)::devBuffer);
}

/*
 * CUDA errors catching block
 */

static void _checkCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define cudaCheck(value) _checkCudaErrorAux(__FILE__, __LINE__, #value, value)

static void _checkCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;

	std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;

	system("pause");

	_free();

	exit(1);
}

/*
 * CUDA kernel block
 */

__global__ void gpuWork(double *data, double *buffer, const std::size_t size,
					    const double stepX, const double stepT, const double maxTime) {
	auto idx = threadIdx.x + blockIdx.x * blockDim.x;

	for (auto i = 0.0; i < maxTime; i += stepT) {
		buffer[size - 1] = data[size - 1] + 5.0;

		if (idx < size - 1 && idx > 0)
			buffer[idx] = ((data[idx + 1] - 2.0 * data[idx] + data[idx - 1]) * stepT / (stepX * stepX)) + data[idx];

		__syncthreads();

		if (idx < size) 
			data[idx] = buffer[idx];

		__syncthreads();
	}
}

__global__ void gpuWorkOptimized(double * __restrict__ data, double * __restrict__ buffer, const std::size_t size,
								 const double stepX, const double stepT, const double maxTime) {
	for (auto i = 0.0; i < maxTime; i += stepT) {
		auto idx = threadIdx.x + blockIdx.x * blockDim.x;

		buffer[size - 1] = data[size - 1] + 5.0;

		if (idx < size - 1 && idx > 0)
			buffer[idx] = ((data[idx + 1] - 2.0 * data[idx] + data[idx - 1]) * stepT / (stepX * stepX)) + data[idx];

		__syncthreads();

		if (idx < size) 
			data[idx] = buffer[idx];

		__syncthreads();
	}
}

/*
 * CPU block
 */

#pragma omp parallel
void cpuWork(double *data, double *buffer, const std::size_t size,
		     const double stepX, const double stepT, const double maxTime) {
#pragma omp for
	for (auto i = 0.0; i < maxTime; i += stepT) {
		buffer[size - 1] = data[size - 1] + 5.0;
#pragma omp for
		for (auto i = 1; i < size - 1; i++)
			buffer[i] = ((data[i + 1] - 2.0 * data[i] + data[i - 1]) * stepT / (stepX * stepX)) + data[i];
		
		std::copy(buffer, buffer + size, data);
	}
}

/*
 * Helpers
 */

int init(std::size_t size) {
	::hostData   = (double *)std::calloc(size, sizeof(double));
	if (!::hostData)
		return 1;

	::hostBuffer = (double *)std::calloc(size, sizeof(double));
	if (!::hostData)
		return 1;

	cudaCheck(cudaMalloc((void **)&::devData, size * sizeof(double)));
	cudaCheck(cudaMalloc((void **)&::devBuffer, size * sizeof(double)));

	std::memset((void *)::hostData, 0, size);
	std::memset((void *)::hostBuffer, 0, size);

	return 0;
}

int gnuplotPrint(const double *result, std::size_t size, double stepX) {
	FILE *gpPipe = nullptr;

#if defined _WIN32
	gpPipe = _popen(GNUPLOT_NAME, "w");
#else
	gpPipe = popen(GNUPLOT_NAME, "w");
#endif

	if (gpPipe == NULL)
		return 1;

	fprintf(gpPipe, "plot '-'\n");

	auto x = 0.0;
	for (auto i = 0; i < size; i++) {
		std::cout << x << " " << hostData[i] << std::endl;

		fprintf(gpPipe, "%f\t%f\n", x, hostData[i]);

		x += stepX;
	}

	std::cout << std::endl;

	fprintf(gpPipe, "%s\n", "e");
	fflush(gpPipe);

	// Waiting for user key input
	std::cin.clear();
	std::cin.ignore(std::cin.rdbuf()->in_avail());
	std::cin.get();

#if defined _WIN32
	_pclose(gpPipe);
#else
	pclose(gpPipe);
#endif

	return 0;
}

int filePrint(const char *filename, const double *result, const std::size_t size, double stepX) {
	std::ofstream ofs(filename, std::ios_base::out | std::ios_base::trunc);

	if (!ofs.is_open())
		return 1;

	ofs << "plot '-'" << std::endl;

	auto x = 0.0;
	for (auto i = 0; i < size; i++) {
		ofs << x << "\t" << result[i] << std::endl;
		x += stepX;
	}
	ofs << "e" << std::endl;

	return 0;
}

/*
 * Main
 */

int main() {
	const size_t length = LENGTH;
	const double time = TIME;

	const double stepX = STEP_X; // Length (x coord) increment
	const double stepT = STEP_T; // Time increment

	const std::size_t nPoints = static_cast<std::size_t>(length / stepX);
	const std::size_t size    = nPoints * sizeof(double);

	if (init(nPoints)) {
		_free();
		return 1;
	}

	cudaCheck(cudaMemcpy(devData, hostData, size, cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(devBuffer, hostBuffer, size, cudaMemcpyHostToDevice));

	//timers

	float timeCPU, timeGPU, timeGPUOpt;
	cudaEvent_t start, stop;

	/*
	 * CPU text
	 */

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	auto beginTime = std::chrono::steady_clock::now();
	cudaEventRecord(start, 0);
	cpuWork(hostData, hostBuffer, nPoints, stepX, stepT, time);
	auto chronoTimeCPU = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - beginTime).count();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&timeCPU, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	if (filePrint("cpu_plot.txt", hostData, nPoints, stepX)) {
		_free();
		return 1;
	}

	/*
	 * GPU test 
	 */

	dim3 nThreads(256);
	dim3 nBlocks(1);

	/*
	 * Default kernel function
	 */

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	beginTime = std::chrono::steady_clock::now();
	cudaEventRecord(start, 0);
	gpuWork <<<nBlocks, nThreads>>> (devData, devBuffer, nPoints, stepX, stepT, time);
	auto chronoTimeGPU = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - beginTime).count();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&timeGPU, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Result to array
	cudaCheck(cudaMemcpy(hostData, devData, size, cudaMemcpyDeviceToHost));

	/*
	 * Kernel function optimization
	 */

	cudaFuncSetCacheConfig(gpuWorkOptimized, cudaFuncCachePreferL1);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	beginTime = std::chrono::steady_clock::now();
	cudaEventRecord(start, 0);
	gpuWorkOptimized <<<nBlocks, nThreads>>> (devData, devBuffer, nPoints, stepX, stepT, time);
	auto chronoTimeGPUOp = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - beginTime).count();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&timeGPUOpt, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	/*
	 * Output
	 */
	
	for (auto i = 0; i < nPoints; i++)
		std::cout << hostData[i] << " ";

	std::cout << std::endl << std::endl;

	/*if (gnuplotPrint(hostData, nPoints, stepX)) {
		_free();
		return 1;
	}*/
	
	if (filePrint("gpu_plot.txt", hostData, nPoints, stepX)) {
		_free();
		return 1;
	}

	std::cout << std::setw(20) << std::left << "CPU time "			 << timeCPU	   << "(" << chronoTimeCPU   << " us by chrono) ms" << std::endl;
	std::cout << std::setw(20) << std::left << "GPU time "			 << timeGPU	   << "(" << chronoTimeGPU   << " us by chrono) ms" << std::endl;
	std::cout << std::setw(20) << std::left << "GPU optimized time " << timeGPUOpt << "(" << chronoTimeGPUOp << " us by chrono) ms" << std::endl;

	/*
	 * Memory free
	 */

	cudaCheck(cudaFree((void *)devData));
	cudaCheck(cudaFree((void *)devBuffer));

	_free();

	system("pause");

	return 0;
}
