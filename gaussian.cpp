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
  unsigned char* out_gaussian = new unsigned char[ width*height*3 ];
  
  unsigned int i, j, c;
  // Initialiser le chronomètre pour avoir le temps d'exécution
std::chrono::time_point<std::chrono::system_clock> start, end;
start = std::chrono::system_clock::now();

  for( j = 2 ; j < height ; ++j ){
  	for ( i = 2 ; i < width ; ++i){
	    for (int k = 0; k < 3; ++k){
	    	out_gaussian[ (width * height * 3) - (3 * (j * width - i)) + k] = 
	    	/* (data[ 3 * ((j-1)*width + (i - 1)) + k] + 2*data[ 3 * ((j-1)*width + i) + k] + data[ 3 * ((j-1)*width + (i + 1)) + k]
	    	+ 2*data[ 3 * ((j)*width + (i - 1)) + k] + 4*data[ 3 * (((j)*width + i)) + k]   + 2*data[ 3 * ((j)*width + (i + 1)) + k]
	    	+ data[ 3 * ((j+1)*width + (i - 1)) + k] + 2*data[ 3 * ((j+1)*width + i) + k] + data[ 3 * ((j+1)*width + (i + 1)) + k]
	    	)/16;
	    	*/
	    	(data[3*((j-2)*width+i-2)+k] + 4*  data[3*((j-2)*width+i-1)+k] + 6*  data[3*((j-2)*width+i)+k] + 4*  data[3*((j-2)*width+i+1)+k] +     data[3*((j-2)*width+i+2)+k] +
            4 * data[3*((j-1)*width+i-2)+k] + 16* data[3*((j-1)*width+i-1)+k] + 24* data[3*((j-1)*width+i)+k] + 16* data[3*((j-1)*width+i+1)+k] + 4*  data[3*((j-1)*width+i+2)+k] +
            6 * data[3*(j*width+i-2)+k]     + 24* data[3*(j*width+i-1)+k]     + 36* data[3*(j*width+i)+k]     + 24* data[3*(j*width+i+1)+k]     + 6*  data[3*(j*width+i+2)+k] +
            4 * data[3*((j+1)*width+i-2)+k] + 16* data[3*((j+1)*width+i-1)+k] + 24* data[3*((j+1)*width+i)+k] + 16* data[3*((j+1)*width+i+1)+k] + 4*  data[3*((j+1)*width+i+2)+k] +
                data[3*((j+2)*width+i-2)+k] + 4*  data[3*((j+2)*width+i-1)+k] + 6*  data[3*((j+2)*width+i)+k] + 4*  data[3*((j+2)*width+i+1)+k] +     data[3*((j+2)*width+i+2)+k]
                )/256;
	    }
	    
    }
  }
  // Afficher le temps d'exécution du programme
end = std::chrono::system_clock::now();
std::chrono::duration<double> elapsed_seconds = end - start;
std::cout << "temps : " << elapsed_seconds.count() << " s" << std::endl;

 
  

  //Placement des données dans l'image
  ilTexImage( width, height, 1, 3, IL_LUMINANCE, IL_UNSIGNED_BYTE, out_gaussian );


  // Sauvegarde de l'image


  ilEnable(IL_FILE_OVERWRITE);

  ilSaveImage("outGaussian.jpg");

  ilDeleteImages(1, &image); 

  delete [] out_gaussian;

}
