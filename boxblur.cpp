#include <iostream>

#include <cmath>

#include <chrono>

#include <IL/il.h>


int main() {

  std::chrono::time_point<std::chrono::system_clock> start, end;

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
  unsigned char* in_red = new unsigned char[ width * height ];
  unsigned char* in_green = new unsigned char[ width * height ];
  unsigned char* in_blue = new unsigned char[ width * height ];
  unsigned char* out_boxblur = new unsigned char[ width * height * 3 ];
  
  unsigned int i, j, c;

  int h, v, res, out_red, out_green, out_blue;

  start = std::chrono::system_clock::now();

  //On sépare les trois composantes des couleurs
  for( std::size_t i = 0 ; i < width*height ; ++i ) {
    in_red[ i ] = data[ 3 * i ];
		in_green[ i ] = data[ 3 * i + 1 ];
		in_blue[ i ] = data[ 3 * i + 2 ];		       
  }

  for(j = 1 ; j < height - 1 ; ++j) {

    for(i = 1 ; i < width - 1 ; ++i) {
      
      if( i > 1 && i < width  && j > 1 && j < height ) {

        out_red = in_red[((j - 1) * width + i - 1) ] + in_red[(j * width + i - 1)] + in_red[((j + 1) * width + i - 1)]
	        +  in_red[((j - 1) * width + i    ) ] + in_red[(j * width + i)] + in_red[((j + 1) * width + i )]
	        +  in_red[((j - 1) * width + i + 1) ] + in_red[(j * width + i + 1 )] + in_red[((j + 1) * width + i + 1) ]; 

        out_red = out_red / 9;

        out_green = in_green[((j - 1) * width + i - 1) ] + in_green[(j * width + i - 1)] + in_green[((j + 1) * width + i - 1)]
	        +  in_green[((j - 1) * width + i    ) ] + in_green[(j * width + i)] + in_green[((j + 1) * width + i )]
	        +  in_green[((j - 1) * width + i + 1) ] + in_green[(j * width + i + 1 )] + in_green[((j + 1) * width + i + 1) ]; 

        out_green = out_green / 9;

        out_blue = in_blue[((j - 1) * width + i - 1) ] + in_blue[(j * width + i - 1)] + in_blue[((j + 1) * width + i - 1)]
	        +  in_blue[((j - 1) * width + i    ) ] + in_blue[(j * width + i)] + in_blue[((j + 1) * width + i )]
	        +  in_blue[((j - 1) * width + i + 1) ] + in_blue[(j * width + i + 1 )] + in_blue[((j + 1) * width + i + 1) ]; 

        out_blue = out_blue / 9;

      }
      else {

        out_red = 0;
        out_green = 0;
        out_blue = 0;

      } 

      out_boxblur[ ( width * height * 3 ) - (3 * (j * width - i))] = out_red;
      out_boxblur[ ( width * height * 3 ) - (3 * (j * width - i)) + 1 ] = out_green;
      out_boxblur[ ( width * height * 3 ) - (3 * (j * width - i)) + 2 ] = out_blue; 
      
    }

  }

  end = std::chrono::system_clock::now();

  //Placement des données dans l'image
  ilTexImage( width , height, 1, 3, IL_LUMINANCE, IL_UNSIGNED_BYTE, out_boxblur );


  // Sauvegarde de l'image


  ilEnable(IL_FILE_OVERWRITE);

  ilSaveImage("outBoxblur.jpg");

  ilDeleteImages(1, &image); 

  delete [] out_boxblur;
  delete [] in_red; 
  delete [] in_green; 
  delete [] in_blue; 

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "temps : " << elapsed_seconds.count() << " s" << std::endl;

}
