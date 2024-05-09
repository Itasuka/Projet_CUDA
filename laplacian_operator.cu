#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>

/**
 * Kernel pour transformer l'image RGB en niveaux de gris.
 */
__global__ void grayscale( unsigned char * rgb, unsigned char * g, std::size_t cols, std::size_t rows ) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;
  if( i < cols && j < rows ) {
    g[ j * cols + i ] = (
			 307 * rgb[ 3 * ( j * cols + i ) ]
			 + 604 * rgb[ 3 * ( j * cols + i ) + 1 ]
			 + 113 * rgb[  3 * ( j * cols + i ) + 2 ]
			 ) >> 10;
  }
}

/**
 * Kernel pour obtenir les contours à partir de l'image en niveaux de gris.
 */
__global__ void laplacian_operator( unsigned char * g, unsigned char * s, std::size_t cols, std::size_t rows )
{
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;

  if( i > 1 && i < cols && j > 1 && j < rows )
  {
    auto res = - g[ (j-1)*cols + i - 1 ] - g[ (j-1)*cols + i ]   - g[ (j-1)*cols + i + 1 ]
               - g[ (j)*cols + i - 1 ]   + 8 * g[ (j)*cols + i ] - g[ (j)*cols + i + 1 ] 
               - g[ (j+1)*cols + i - 1 ] - g[ (j+1)*cols + i ]   - g[ (j+1)*cols + i + 1 ];

    res = res > 128 ? res : 0;
    s[ j * cols + i ] = res;
  }
}


int main()
{
  cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );

  //auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;

  //std::vector< unsigned char > g( rows * cols );
  // Allocation de l'image de sortie en RAM côté CPU.
  unsigned char * g = nullptr;
  cudaMallocHost( &g, rows * cols );
  cv::Mat m_out( rows, cols, CV_8UC1, g );

  // Copie de l'image en entrée dans une mémoire dite "pinned" de manière à accélérer les transferts.
  // OpenCV alloue la mémoire en interne lors de la décompression de l'image donc soit sans doute avec
  // un malloc standard.
  unsigned char * rgb = nullptr;
  cudaMallocHost( &rgb, 3 * rows * cols );
  
  std::memcpy( rgb, m_in.data, 3 * rows * cols );

  unsigned char * rgb_d;
  unsigned char * g_d;
  unsigned char * s_d;

  cudaMalloc( &rgb_d, 3 * rows * cols );
  cudaMalloc( &g_d, rows * cols );
  cudaMalloc( &s_d, rows * cols );

  cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );

  dim3 block( 32, 16 );
  dim3 grid0( ( cols - 1) / block.x + 1 , ( rows - 1 ) / block.y + 1 );
    
  cudaEvent_t start, stop;

  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  // Mesure du temps de calcul du kernel uniquement.
  grayscale<<< grid0, block >>>( rgb_d, g_d, cols, rows );
  cudaEventRecord( start );

  laplacian_operator<<< grid0, block >>>( g_d, s_d, cols, rows );  

  
  cudaEventRecord( stop );
  
  cudaMemcpy( g, s_d, rows * cols, cudaMemcpyDeviceToHost );

  cudaEventSynchronize( stop );
  float duration;
  cudaEventElapsedTime( &duration, start, stop );
  std::cout << "time=" << duration << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cv::imwrite( "outLaplacian_cu.jpg", m_out );

  cudaFree( rgb_d);
  cudaFree( g_d);
  cudaFree( s_d);

  cudaFreeHost( g );
  cudaFreeHost( rgb );
  
  return 0;
}
