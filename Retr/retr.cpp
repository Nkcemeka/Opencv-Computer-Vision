#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(){
    /* Contour detection and drawing using different extraction modes to complement the understanding of 
    hierarchies.
    */

   Mat image = imread("../../img.png");
   Mat img_gray;

   cvtColor(image, img_gray, COLOR_BGR2GRAY);
   Mat thresh;
   threshold(img_gray, thresh, 150, 255, THRESH_BINARY);

   // RETR_LIST retrieval mode: Does not create any parent child relationship for the extracted contours.
//    vector<vector<Point>> contours;
//    vector<Vec4i> hierarchy;
//    findContours(thresh, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);
//    Mat image_copy = image.clone();
//    drawContours(image_copy, contours, -1, Scalar(255,0,0), 2);
//    imshow("LIST", image_copy);
//    waitKey(0);
//    imwrite("../../contours_retr_list.jpg", image_copy);

   // RETR_EXTERNAL retrieval mode: Detects the parent contours and ignores any child contours
   vector<vector<Point>> contours2;
   vector<Vec4i> hierarchy2;
   findContours(thresh, contours2, hierarchy2, RETR_EXTERNAL, CHAIN_APPROX_NONE);
   Mat image_copy2 = image.clone();
   drawContours(image_copy2, contours2, -1, Scalar(255,0,0), 2);
   imshow("EXTERNAL", image_copy2);
   waitKey(0);
   imwrite("../../contours_retr_external.jpg", image_copy2);
}