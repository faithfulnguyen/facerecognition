#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <dirent.h>
#include <iostream>
#include <vector>
#include <string>

namespace faceRecog{
	using namespace std;
	using namespace cv;
	using namespace cv::ml;
	class FaceRecognition{
	private:
		vector <string> pathDatasTestFAR;
		vector <string> pathDatasTestFRR;
		vector <string> pathDatasTrain;
		vector <Mat> faceTrain;
		vector <Mat> faceTestFAR;
		vector <Mat> faceTestFRR;
		vector <Mat> dataSVM;

		string pathDBTrain;
		string pathDBTestFAR;

		PCA pca;
		Ptr <SVM> svm;
		CascadeClassifier face_cascade;

		int numPCA;
		int faceHight;
		int faceWidth;

	public:
		FaceRecognition(){
			this->face_cascade.load("./haarcascades/haarcascade_frontalface_default.xml");
			this->pathDBTrain = "./Data/";
			this->pathDBTestFAR = "./";
			this->faceHight = 256;
			this->faceWidth = 246;
			this->numPCA = 51;
		}
		//API
		void readDBPathTrain();
		void readDBPathTestFAR();
		bool readDBpath(string path, vector<string>& pathTrain, vector<string>& pathTest, int kFold);

		// API
		void readImageTrain();
		void readImageTestFAR();
		void readImageTestFRR();
		//
		bool readImage(vector<string> pathDB, vector<Mat>& face );
		Mat faceDetector(Mat image);
		bool doPCA();
		bool extractFeatureForTrain();
		Mat extractEigenVector(Mat image);
		bool trainSVM();

	};
}
