#include <iostream>

#include <cmath>

#include <IL/il.h>

#include <chrono>


int main() {

  unsigned int image;

  ilInit();

  ilGenImages(1, &image);
  ilBindImage(image);
  ilLoadImage("in.jpg");

  int width, height, bpp, format;

  width = ilGetInteger(IL_IMAGE_WIDTH);
  height = ilGetInteger(IL_IMAGE_HEIGHT); 
  bpp = ilGetInteger(IL_IMAGE_BYTES_PER_PIXEL);
  format = ilGetInteger(IL_IMAGE_FORMAT);

  // Récupération des données de l'image
  unsigned char* data = ilGetData();

  // Traitement de l'image
  unsigned char* out_grey = new unsigned char[ width*height ];
  unsigned char* out_laplacian = new unsigned char[width*height ];
  
  // Initialiser le chronomètre pour avoir le temps d'exécution
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  for( std::size_t i = 0 ; i < width*height ; ++i )
  {
    // GREY = ( 307 * R + 604 * G + 113 * B ) / 1024
    out_grey[ i ] = ( 307 * data[ 3*i ]
		       + 604 * data[ 3*i+1 ]
		       + 113 * data[ 3*i+2 ]
		       ) >> 10;
  }
  
  unsigned int i, j, c;

  int h, v, res;


  for(j = 1 ; j < height - 1 ; ++j) {

    for(i = 1 ; i < width - 1 ; ++i) {
    
	    auto res = - out_grey[ (j-1)*width + i - 1 ] - out_grey[ (j-1)*width + i ]   - out_grey[ (j-1)*width + i + 1 ]
		       - out_grey[ (j)*width + i - 1 ]   + 8 * out_grey[ (j)*width + i ] - out_grey[ (j)*width + i + 1 ] 
		       - out_grey[ (j+1)*width + i - 1 ] - out_grey[ (j+1)*width + i ]   - out_grey[ (j+1)*width + i + 1 ];

	    res = res > 128 ? res : 0;
	    out_laplacian[ (width * height) - j * width + i ] = res;

    }

  }
  
  // Afficher le temps d'exécution du programme
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "temps : " << elapsed_seconds.count() << " s" << std::endl;

  //Placement des données dans l'image
  ilTexImage( width, height, 1, 1, IL_LUMINANCE, IL_UNSIGNED_BYTE, out_laplacian );


  // Sauvegarde de l'image


  ilEnable(IL_FILE_OVERWRITE);

  ilSaveImage("outLaplacian.jpg");

  ilDeleteImages(1, &image); 

  delete [] out_grey;
  delete [] out_laplacian;

}
