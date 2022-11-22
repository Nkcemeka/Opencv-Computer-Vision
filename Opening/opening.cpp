// OPENING MORPHOLOGICAL OPERATION

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

// Function to perform Morphological Opening
Mat performOpening(Mat inputImage, int morphologyElement, int erosionSize, int dilationSize){
    Mat outputImage, tempImage, mid;
    int morphologyType;

    if (morphologyElement == 0){
        morphologyType = MORPH_RECT;
    } else if (morphologyElement == 1){
        morphologyType = MORPH_CROSS;
    } else if (morphologyElement == 2){
        morphologyType = MORPH_ELLIPSE;
    }

    // Create the structuring element for opening
    Mat erosionElement = getStructuringElement(morphologyType, Size(2*erosionSize+1, 2*erosionSize+1), Point(erosionSize, erosionSize));
    Mat dilationElement = getStructuringElement(morphologyType, Size(2*dilationSize+1, 2*dilationSize+1), Point(dilationSize, dilationSize));

    // Apply morphological opening to the image using the structuring element.
    erode(inputImage, tempImage, erosionElement);
    dilate(tempImage, outputImage, dilationElement);


    return outputImage;
}

int main(){
    Mat inputImage, outputImage, thresh, numObject;
    int morphologyElement = 0;
    int erosionSize = 2;
    int dilationSize = 2;

    inputImage = imread("../../salt.png");

    if (inputImage.channels() != 1){
        cvtColor(inputImage, inputImage, COLOR_BGR2GRAY);
    }

    if (!inputImage.data){
        cout << "Invalid input image. Exiting!" << endl;
        return -1;
    }


    // Apply Morphological opening
    outputImage = performOpening(inputImage, morphologyElement, erosionSize, dilationSize);

    // Create Window to display output image
    namedWindow("Input image", WINDOW_AUTOSIZE);
    namedWindow("Output image after opening", WINDOW_AUTOSIZE);

    // Display output
    imshow("Input image", inputImage);
    imshow("Output image after opening", outputImage);

    // Wait until the user hits a key on the keyboard
    waitKey(0);

    return 0;
}