#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/face.hpp>
#include <sstream>
#include <fstream>
#include <sys/resource.h>
#include <dirent.h>
#include <vector>
#include <string>
#include <time.h>

class FaceRecognition{
private:
	std :: string pathDataTrain;
	std :: string pathDataTestFAR;

	std :: vector <cv :: Mat> faceTrain;
	std :: vector <cv :: Mat> faceTestFRR;
	std :: vector <cv :: Mat> faceTestFAR;

	std :: vector <std :: string> fileDataTrain;
	std :: vector <std :: string> fileDataTestFRR;
	std :: vector <std :: string> fileDataTestFAR;
	std :: vector <cv :: Mat> dataSVM;
	std :: vector<std :: string> namePeoples;


	cv :: PCA pca;
	cv :: Ptr<cv :: ml ::SVM> svm;
	cv :: CascadeClassifier face_cascade;

	int faceHight;
	int faceWidth;
	int numPCA;
public:
	FaceRecognition(){
	    pathDataTrain = "./Face_Database/Train";
		pathDataTestFAR =  "./Face_Database/Test";

		face_cascade.load("./haarcascades/haarcascade_frontalface_default.xml");
		this->svm = cv :: ml :: SVM::create();
		svm->setType(cv ::ml :: SVM::C_SVC);
		//the last C = 3
		svm->setC(55);
		svm->setKernel(cv :: ml :: SVM::RBF);
		svm->setTermCriteria(cv :: TermCriteria(cv :: TermCriteria::MAX_ITER, (int)1e7, 1e-6));
		faceHight = 64;
		faceWidth = 64;
		numPCA = 35;
		this->loadNamePeoples();
	}


	//API
	void readDataTrain();
	void readDataTestFAR();
	// code detail
	void readPathData( std :: string pathData, std :: vector <std :: string>& fileTrain, std :: vector <std :: string>& fileTest, int number);

	//API
	void readImageTrain();
	void readImageTestFRR();
	void readImageTestFAR();
	void readImage(std :: vector<cv :: Mat>& faceHuman, std :: vector<std::string>pathData);

	void doPCA();
	void faceRecognitionTrain();
	void vectorPCA();
	void convertToMatrix();
	void predictFAR();
	void predictFRR();
	void prediction(std :: vector <cv :: Mat> dataTest, cv :: Mat lb);


	void trainSVM(cv :: Mat label, std :: string model);
	void loadModel(std :: string model);

	void loadNamePeoples();

	cv :: Mat faceRecognitionCreateLabelTrain();
	cv :: Mat faceRecognitionCreateLabelTestFRR();
	cv :: Mat faceRecognitionCreateLabelTestFAR();
	cv :: Mat faceDetector( cv :: Mat image );


	~ FaceRecognition(){
		//std :: cout << "destroy FaceRecognition object" << std :: endl;
	}

};
