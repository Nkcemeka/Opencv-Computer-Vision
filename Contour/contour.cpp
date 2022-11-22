#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(){
    Mat inputImage, threshImage;
    inputImage = imread("../../img.png");

    if (inputImage.empty()){
        cout << "Input image is empty! Exiting!";
        return 0;
    }

    cvtColor(inputImage, threshImage, COLOR_BGR2GRAY); // Converting input image to a grayscale image

    threshold(threshImage, threshImage, 150, 255, THRESH_BINARY);

    // Detect the contours on the binary image using CHAIN_APPROX_NONE
    vector<vector<Point>> contours; // Contour is a vector of point vectors. Each point vector contains many coordinates that identify a particular contour
    vector<Vec4i> hierarchy;
    findContours(threshImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
    

    // Draw contours on the original image
    Mat image_copy = inputImage.clone();
    drawContours(image_copy, contours, -1, Scalar(0,255,0), 2); // -1 means show all contour; 0,255,0 means show contours in green; 2 is contour thickness
    //imshow("None Approximation", image_copy);
    //waitKey(0);
    //imwrite("../../contours_none_image1.jpg", image_copy);


    // Code for CHAIN_APPROX_SIMPLE
    Mat image_copy2 = inputImage.clone();
    vector<vector<Point>> contours2;
    vector<Vec4i> hierarchy2;
    findContours(threshImage, contours2, hierarchy2, RETR_TREE, CHAIN_APPROX_NONE);

    drawContours(image_copy2, contours2, -1, Scalar(255,0,0), 2);
    for (int i = 0; i<contours2.size(); i = i+1){
        cout << contours2[i] << endl;
        for (int j = 0; j<contours2[i].size(); j=j+1){
            cout << contours2[i][0] << " " << contours2[i][1] << endl;
            circle(image_copy2, (contours2[i][0], contours2[i][1]), 2, Scalar(0,255,0), 2);
        }
    }

    imshow("CHAIN_APPROX_SIMPLE Point only", image_copy2);
    waitKey(0);
}



