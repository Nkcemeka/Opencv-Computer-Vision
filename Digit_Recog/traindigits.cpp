#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

string pathName = "digits.png";
int SZ = 20; // SZ is size of each digit in the image
float affineFlags = WARP_INVERSE_MAP|INTER_LINEAR;

Mat deskew(Mat& img){
    Moments m = moments(img);

    // if this is true, no need to deskew
    if (abs(m.mu02) < 1e-2){
        return img.clone();
    }

    float skew = m.mu11/m.mu02;
    Mat warpMat = (Mat_<float>(2,3) << 1, skew, -0.5*SZ*skew, 0, 1, 0);
    Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());
    warpAffine(img, imgOut, warpMat, imgOut.size(), affineFlags);

    return imgOut;
}


void loadTrainTestLabel(string &pathName, vector<Mat> &trainCells, vector<Mat>& testCells, vector<int> &trainLabels, vector<int> &testLabels){
    Mat img = imread(pathName, IMREAD_GRAYSCALE);
    int ImgCount = 0;
    for (int i = 0; i < img.rows; i = i + SZ){
        for (int j =0; j < img.cols; j = j + SZ){
            Mat digitImg = (img.colRange(j, j+SZ).rowRange(i, i+SZ)).clone(); // extract each digit from the overall image
            if (j < int(0.9*img.cols)){
                trainCells.push_back(digitImg); // This selects 90% of the digits as training data
            }else{
                testCells.push_back(digitImg); // This selects 10% of the digits as test data
            }
            ImgCount++;
        }
    }

    cout << "Image Count: " << ImgCount << endl;
    float digitClassNumber = 0;

    // Sets 90% of the digits to their respective values in trainLabels
    for (int z = 0; z < int(0.9*ImgCount); z++){
        if (z%450 == 0 && z!=0){
            digitClassNumber = digitClassNumber + 1;
        }
        trainLabels.push_back(digitClassNumber);
    }

    digitClassNumber = 0;

    // Sets 10% of the digits to their respective values in testLabels
    for (int z = 0; z < int(0.1*ImgCount); z++){
        if (z%50 == 0 && z!=0){
            digitClassNumber = digitClassNumber + 1;
        }
        testLabels.push_back(digitClassNumber);
    }
}

void CreateDeskewedTrainTest(vector<Mat> &deskewedTrainCells, vector<Mat>& deskewedTestCells, vector<Mat>& trainCells, vector<Mat>& testCells){
    for (int i = 0; i < trainCells.size(); i++){
        Mat deskewedImg = deskew(trainCells[i]);
        deskewedTrainCells.push_back(deskewedImg);
    }

    for (int i = 0; i < testCells.size(); i++){
        Mat deskewedImg = deskew(testCells[i]);
        deskewedTestCells.push_back(deskewedImg);
    }
}

HOGDescriptor hog(
    Size(20,20), // WinSize
    Size(8,8), // blockSize
    Size(4,4), // blockStride
    Size(8,8), // cellSize
    9, // nbins
    1, // derivAper
    -1, // winSigma
    cv::HOGDescriptor::HistogramNormType::L2Hys, // histogramNormType
    0.2, //L2HYSThresh
    0, // gammal correction
    64, // nlevels = 64
    1 // Use signed gradients
); 

void createTrainTestHOG(vector<vector<float>>& trainHOG, vector<vector<float>>& testHOG, vector<Mat> &deskewedtrainCells, vector<Mat> &deskewedtestCells){
    for (int y = 0; y < deskewedtrainCells.size(); y++){
        // Gets a HOG feature vector for each of the deskewed images stored in deskewedtrainCells
        vector<float> descriptors;
        hog.compute(deskewedtrainCells[y], descriptors);
        trainHOG.push_back(descriptors);
    }

    for (int y = 0; y < deskewedtestCells.size(); y++){
        // Gets a HOG feature vector for each of the deskewed images stored in deskewedtestCells
        vector<float> descriptors;
        hog.compute(deskewedtestCells[y], descriptors);
        testHOG.push_back(descriptors);
    }
}

void ConvertVectortoMatrix(vector<vector<float>> &trainHOG, vector<vector<float>> &testHOG, Mat &trainMat, Mat &testMat){
    int descriptor_size = trainHOG[0].size();

    // Converts the HOG feature vector for each deskewed training image back to a matrix
    for (int i = 0; i < trainHOG.size(); i++){
        for (int j = 0; j < descriptor_size; j++){
            trainMat.at<float>(i, j) = trainHOG[i][j];
        }
    }

    // Converts the HOG feature vector for each deskewed test image back to a matrix
    for (int i = 0; i < testHOG.size(); i++){
        for (int j = 0; j < descriptor_size; j++){
            testMat.at<float>(i, j) = testHOG[i][j];
        }
    }
}

void getSVMParams(SVM *svm){
    cout << "Kernel type :" << svm -> getKernelType() << endl;
    cout << "Type        :" << svm -> getType() << endl;
    cout << "C           :" << svm -> getC() << endl;
    cout << "Degree      :" << svm -> getDegree() << endl;
    cout << "Nu          :" << svm -> getNu() << endl;
    cout << "Gamma       :" << svm -> getGamma() << endl;
}

Ptr<SVM> svmInit(float C, float gamma){ // associated with cv::ml namespace
    Ptr<SVM> svm = SVM::create();
    svm->setGamma(gamma); // Set parameter gamma
    svm->setC(C); // set C parameter used to classify or separate datasets
    svm->setKernel(SVM::RBF); // set SVM Kernel to Radial Basis function
    svm->setType(SVM::C_SVC); // set SVM type
    return svm;
}

void svmTrain(Ptr<SVM> svm, Mat &trainMat, vector<int> &trainLabels){
    Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
    svm->train(td);
    svm->save("results/eyeGlassClassifierModel.yml");
}

void svmPredict(Ptr<SVM> svm, Mat &testResponse, Mat &testMat){
    svm -> predict(testMat, testResponse);
}

void SVMevaluate(Mat &testResponse, float &count, float &accuracy, vector<int> &testLabels){
    for (int i = 0; i < testResponse.rows; i++){
        if (testResponse.at<float>(i,0) == testLabels[i]){
            count = count + 1;
        }
    }
    accuracy = (count/testResponse.rows)*100;
}

int main(){
    vector<Mat> trainCells;
    vector<Mat> testCells;
    vector<int> trainLabels;
    vector<int> testLabels;
    loadTrainTestLabel(pathName, trainCells, testCells, trainLabels, testLabels);

    vector<Mat> deskewedTrainCells;
    vector<Mat> deskewedTestCells;
    CreateDeskewedTrainTest(deskewedTrainCells, deskewedTestCells, trainCells, testCells);

    std::vector<std::vector<float>> trainHOG;
    std::vector<std::vector<float>> testHOG;
    createTrainTestHOG(trainHOG, testHOG, deskewedTrainCells, deskewedTestCells);

    int descriptor_size = trainHOG[0].size();
    cout << "Descriptor Size: " << descriptor_size << endl;

    Mat trainMat(trainHOG.size(), descriptor_size, CV_32FC1);
    Mat testMat(testHOG.size(), descriptor_size, CV_32FC1);

    ConvertVectortoMatrix(trainHOG, testHOG, trainMat, testMat);

    float C = 12.5, gamma = 0.5;

    Mat testResponse;
    Ptr<SVM> model = svmInit(C, gamma);


    //////////// SVM TRAINING//////////
    svmTrain(model, trainMat, trainLabels);

    //////////// SVM TESTING //////////
    svmPredict(model, testResponse, testMat);

    ////// FIND ACCURACY //////// 
    float count = 0;
    float accuracy = 0;
    getSVMParams(model);
    SVMevaluate(testResponse, count, accuracy, testLabels);

    cout << "the accuracy is: " << accuracy << endl;
    return 0;
}



