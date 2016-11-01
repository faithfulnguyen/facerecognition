#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <dirent.h>
#include <iostream>
#include <vector>
#include <string>
#include "svm_pca.h"


using namespace faceRecog;
int main(){
	FaceRecognition obj;
	obj.readDBPathTrain();
	obj.readImageTrain();
	obj.doPCA();

}
