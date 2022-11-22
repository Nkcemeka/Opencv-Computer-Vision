// #include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/highgui/highgui.hpp"
// #include <iostream>

// using namespace std;
// using namespace cv;

// int main(){
//     Mat inputImage, threshImage;
//     inputImage = imread("../../img.png");

//     if (inputImage.empty()){
//         cout << "Input image is empty! Exiting!";
//         return 0;
//     }

//     cvtColor(inputImage, threshImage, COLOR_BGR2GRAY); // Converting input image to a grayscale image

//     threshold(threshImage, threshImage, 150, 255, THRESH_BINARY);

//     // Detect the contours on the binary image using CHAIN_APPROX_NONE
//     vector<vector<Point>> contours;
//     vector<Vec4i> hierarchy;
//     findContours(threshImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

//     // Draw contours on the original image
//     Mat image_copy = inputImage.clone();
//     drawContours(image_copy, contours, -1, Scalar(0,255,0), 2); // -1 means show all contour; 0,255,0 means show contours in green; 2 is contour thickness
//     imshow("None Approximation", image_copy);
//     waitKey(0);
//     imwrite("../../contours_none_image1.jpg", image_copy);
// }



