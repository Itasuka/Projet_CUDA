#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>

__global__ void boxblur( unsigned char * rgb, unsigned char * rgb2, std::size_t cols, std::size_t rows ) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;
  if( i>2 && i < cols-2 && j>1 && j < rows ) {
  
  for(int k=0; k<3; ++k){
      auto res =     rgb[ 3 * ((j-1) * cols + i-1) + k ]    + rgb[ 3 * ((j-1) * cols + i) + k ]     + rgb[ 3 * ((j-1) * cols + i+1) + k ]   +
                     rgb[ 3 * (j * cols + i-1) + k ]        + rgb[ 3 * (j * cols + i) + k ]         + rgb[ 3 * (j * cols + i+1) + k]        +
                     rgb[ 3 * ((j+1) * cols + i-1) + k ]    + rgb[ 3 * ((j+1) * cols + i) + k]      + rgb[ 3 * ((j+1) * cols + i+1) + k ];
                               
      rgb2[ 3 *(j * cols + i) + k ] = res/9;
  }
  
    
  }  
}

int main()
{
  cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );

  //auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols; 

  // Copie de l'image en entrée dans une mémoire dite "pinned" de manière à accélérer les transferts.
  // OpenCV alloue la mémoire en interne lors de la décompression de l'image donc soit sans doute avec
  // un malloc standard.
  unsigned char * rgb = nullptr;
  cudaMallocHost( &rgb, 3 * rows * cols );
  
  std::memcpy( rgb, m_in.data, 3 * rows * cols );
  
  unsigned char * rgb2 = nullptr;
  cudaMallocHost( &rgb2, 3 * rows * cols );
  
  cv::Mat m_out( rows, cols, CV_8UC3, rgb2 );

  unsigned char * rgb_d;
  unsigned char * rgb2_d;

  cudaMalloc( &rgb_d, 3 * rows * cols );
  cudaMalloc( &rgb2_d, 3 * rows * cols );

  cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );
  cudaMemcpy( rgb2_d, rgb2, 3 * rows * cols, cudaMemcpyHostToDevice );

  dim3 block( 32, 16 );
  dim3 grid0( ( cols  - 1) / block.x + 1 , ( rows - 1 ) / block.y + 1 );
    
  cudaEvent_t start, stop;

  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  // Mesure du temps de calcul du kernel uniquement.
  cudaEventRecord( start );

  
  boxblur<<< grid0, block >>>( rgb_d, rgb2_d, cols, rows * 3 );

  cudaEventRecord( stop );
  
  cudaMemcpy( rgb2, rgb2_d, 3 * rows * cols, cudaMemcpyDeviceToHost );

  cudaEventSynchronize( stop );
  float duration;
  cudaEventElapsedTime( &duration, start, stop );
  std::cout << "time=" << duration << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cv::imwrite( "outBoxblur_cu.jpg", m_out );

  cudaFree( rgb_d);
  cudaFree( rgb2_d);

  cudaFreeHost( rgb );
  cudaFreeHost( rgb2 );
  
  return 0;
}
