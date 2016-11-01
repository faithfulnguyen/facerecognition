#include "svm_pca.h"
using namespace faceRecog;
void FaceRecognition :: readDBPathTrain(){
	this->readDBpath(this->pathDBTrain, this->pathDatasTrain, this->pathDatasTestFRR, this->numPCA);
}
void FaceRecognition :: readDBPathTestFAR(){
	this->readDBpath(this->pathDBTestFAR, this->pathDatasTestFAR, this->pathDatasTestFAR, this->numPCA);
}

bool FaceRecognition :: readDBpath(string path, vector<string>& pathTrain, vector<string>& pathTest, int kFold){
	DIR *dir;
	struct dirent *file;
	if ((dir = opendir(path.c_str())) != NULL) {
	  while ((file = readdir(dir)) != NULL) {
		  DIR *dirChild;
		  struct dirent *entChild;
		  string pathDIR = path + file->d_name + "/";
		  int idx = 0;
		  if ((dirChild = opendir(pathDIR.c_str())) != NULL) {
		  	  while ((entChild = readdir(dirChild)) != NULL){
		  		  string pathImage = pathDIR + entChild->d_name;
		  		  if((pathImage.find(".jpg") != string::npos) || (pathImage.find(".Jpg") != string::npos )){
		  			  if(idx < kFold){
		  				  pathTrain.push_back(pathImage);
		  			  }
		  			  else
		  				  pathTest.push_back(pathImage);
		  			  kFold++;

		  		  	  }
		  	  }
		  }
		  closedir(dirChild);
	  }
	}
	closedir(dir);
	return true;
}


void FaceRecognition :: readImageTrain(){
	this->readImage(this->pathDatasTrain, this->faceTrain);
}
void FaceRecognition :: readImageTestFAR(){
	this->readImage(this->pathDatasTestFAR, this->faceTestFAR);
}
void FaceRecognition :: readImageTestFRR(){
	this->readImage(this->pathDatasTestFRR, this->faceTestFRR);
}
bool FaceRecognition :: readImage(vector<string> pathDB, vector<Mat>& face ){
	if(pathDB.size() < 1){
		cout << "please Check your folder contain data!" << endl;
		return false;
	}
	for(int i = 0; i < pathDB.size(); i++){
		Mat image = imread(pathDB[i], CV_LOAD_IMAGE_GRAYSCALE);
		Mat fDetect = this->faceDetector(image);
		face.push_back(fDetect);
	}
	return true;
}

Mat FaceRecognition :: faceDetector(Mat image){
	Mat crop;
	Mat res;
	Mat gray;
	vector<Rect> faces;
	Mat face;
	face_cascade.detectMultiScale(image, faces, 1.1, 2, 0| CV_HAAR_SCALE_IMAGE, Size(60, 90));

	if(faces.size() < 1){
		resize(image, image, Size(this->faceWidth, this->faceHight), 0, 0 , INTER_LINEAR);
		equalizeHist(image, image);
		return image;
	}
	cv::Rect roi_b;
	cv::Rect roi_c;

	size_t ic = 0; // ic is index of current element
	int ac = 0; // ac is area of current element

	size_t ib = 0; // ib is index of biggest element
	int ab = 0; // ab is area of biggest element

	for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)
	{
		roi_c.x = faces[ic].x;
		roi_c.y = faces[ic].y;
		roi_c.width = (faces[ic].width);
		roi_c.height = (faces[ic].height);

		ac = roi_c.width * roi_c.height; // Get the area of current element (detected face)

		roi_b.x = faces[ib].x;
		roi_b.y = faces[ib].y;
		roi_b.width = (faces[ib].width);
		roi_b.height = (faces[ib].height);

		ab = roi_b.width * roi_b.height; // Get the area of biggest element, at beginning it is same as "current" element

		if (ac > ab)
		{
			ib = ic;
			roi_b.x = faces[ib].x;
			roi_b.y = faces[ib].y;
			roi_b.width = (faces[ib].width);
			roi_b.height = (faces[ib].height);
		}

		crop = image(roi_b);
		resize(crop, res, Size(this->faceWidth, this->faceHight), 0, 0, INTER_LINEAR);
	}
	equalizeHist(crop, crop);
	return crop;
}

bool FaceRecognition :: doPCA(){
	cout << "calculate PCA" << endl;
	int total = this->faceTrain[0].rows * this->faceTrain[0].cols;
	Mat mat(total, this->faceTrain.size(), CV_32FC1);
	for(int i = 0; i < this->faceTrain.size(); i++) {
		Mat X = mat.col(i);
		faceTrain[i].reshape(1, total).col(0).convertTo(X, CV_32FC1, 1/255.);
	}
	this->pca(mat, Mat(), CV_PCA_DATA_AS_COL, this->numPCA);
	return true;
}

Mat FaceRecognition :: extractEigenVector(Mat image){
	Mat tmp = image.reshape(1, image.rows * image.cols);
	Mat egVector(1, this->numPCA, CV_32FC1);
	egVector = this->pca.project(tmp);
	normalize(egVector, egVector);
	return egVector;
}

bool FaceRecognition :: extractFeatureForTrain(){
	for(int i = 0; i < this->faceTrain.size(); i++){
		Mat tmp(1, this->numPCA, CV_32FC1);
		tmp = extractEigenVector(faceTrain[i]);
		this->dataSVM.push_back(tmp);
	}
	return true;
}
bool FaceRecognition :: trainSVM(){
	return true;
}







