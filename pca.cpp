#include "pca.h"

#define TOTAL
#define MAXPEOPLE 15
#define MAX_IMAGE_TRAIN 20
#define TOTAl_IMAGE 74
void FaceRecognition :: readDataTrain(){
	std :: cout << "read data train!" << std :: endl;
	readPathData(this->pathDataTrain, this->fileDataTrain, this->fileDataTestFRR, MAX_IMAGE_TRAIN);
	std :: cout << "read data test FRR!" << std :: endl;
}

void FaceRecognition :: readDataTestFAR(){
	std :: cout << "read Data test FAR!" << std :: endl;
	readPathData(this->pathDataTestFAR, this->fileDataTestFAR, this->fileDataTestFAR, MAX_IMAGE_TRAIN);
}

void FaceRecognition :: readPathData(std :: string pathData, std :: vector <std :: string>& fileTrain, std :: vector <std :: string>& fileTest, int number){
	DIR *imgDp;
	struct dirent *dbDirp;
	imgDp = opendir(pathData.c_str());
	int tl = 0;
	while ((dbDirp = readdir(imgDp))){
		std :: string userID = dbDirp->d_name;
		std :: string userDirectory = pathData + "/" + userID;
		if (strstr(userDirectory.c_str(), "/.") != NULL || strstr(userDirectory.c_str(), "/..") != NULL) {
			continue;
		}
		DIR *userDp;
		struct dirent *dataDirp;
		userDp = opendir(userDirectory.c_str());
		int index = 0;
		while ((dataDirp = readdir(userDp))) {
			std :: string fileName = userDirectory + "/" + dataDirp->d_name;
			if (strstr(fileName.c_str(), ".jpg") != NULL || strstr(fileName.c_str(), ".Jpg") != NULL) {
				;
			}else {
				continue;
			}
			if(index < number){
				fileTrain.push_back(fileName);
			}
			else {
				fileTest.push_back(fileName);
			}
			index++;
			tl++;
		}
		closedir(userDp);
	}
	closedir(imgDp);
}

void FaceRecognition :: readImageTrain(){
	std :: cout << "Detect face data Train!" << std :: endl;
	readImage(this->faceTrain, this->fileDataTrain);
}

void FaceRecognition :: readImageTestFRR(){
	std :: cout << "Detect face data Test FRR!" << std :: endl;
	readImage(this->faceTestFRR, this->fileDataTestFRR);
}

void FaceRecognition :: readImageTestFAR(){
	std :: cout << "Detect face data test FAR!" << std :: endl;
	readImage(this->faceTestFAR, this->fileDataTestFAR);
}

void FaceRecognition :: readImage(std :: vector<cv :: Mat>& faceHuman, std :: vector<std::string>pathData){

	for(int i = 0; i < pathData.size(); i++){
		cv :: Mat image = cv :: imread(pathData[i], CV_LOAD_IMAGE_GRAYSCALE);
		cv :: Mat face = this->faceDetector(image);
		faceHuman.push_back(face);
	}
}

cv :: Mat FaceRecognition :: faceDetector(cv :: Mat image){
	std :: vector<cv :: Rect> objectList;
	this->face_cascade.detectMultiScale(image, objectList, 1.1, 3, 0| CV_HAAR_SCALE_IMAGE, cv :: Size(60,90));
	cv :: Mat face_resized;
	if (objectList.size() <= 0 ){
		resize (image, image, cv :: Size(this->faceHight, this->faceWidth));
		return image;
	}
	//Only support biggest face
	cv :: Rect maxRect;
	for (size_t i = 0; i < objectList.size(); i++){
		if (objectList[i].area() > maxRect.area()){
			maxRect = objectList[i];
		}
	}

	maxRect.x = maxRect.x + maxRect.width * 0.1;
	maxRect.width = maxRect.width - maxRect.width * 0.2;

	cv :: Mat face = image(maxRect);


	if (maxRect.width < this->faceWidth && maxRect.height < this->faceHight){
		cv :: resize(face, face_resized, cv :: Size(this->faceWidth, this->faceHight), CV_INTER_LINEAR);
	}
	else{
		cv :: resize(face, face_resized, cv :: Size(this->faceWidth, this->faceHight), CV_INTER_AREA);
	}
	cv :: equalizeHist(face_resized, face_resized);
	cv :: GaussianBlur( face_resized, face_resized, cv :: Size( 3, 3 ), 0, 0 );
	cv :: imshow("face", face_resized);
	cv :: waitKey(100);
	return face_resized;
}

// convert to mat for PCA
void FaceRecognition :: convertToMatrix(){
	std :: cout << "Convert Data to Matrix!" << std :: endl;
	int total = 0;
	total = this->faceTrain[0].rows * this->faceTrain[0].cols;
	cv :: Mat mat(total, this->faceTrain.size(), CV_32FC1);
	for(int i = 0; i < this->faceTrain.size(); i++) {
		cv :: Mat X = mat.col(i);
		faceTrain[i].reshape(1, total).col(0).convertTo(X, CV_32FC1, 1/255.);
	}
	std :: cout << "starting calculate PCA!" << std :: endl;
	this->pca(mat, cv :: Mat(), CV_PCA_DATA_AS_COL, this->numPCA);
}

void FaceRecognition :: doPCA(){
	clock_t tStart = clock();
	convertToMatrix();
	std :: cout << "Extract eigen vector." << std :: endl;
	vectorPCA();
	std :: cout << "time: " <<(double)(clock() - tStart)/CLOCKS_PER_SEC << std :: endl;
}

void FaceRecognition :: vectorPCA(){
	int total = 0;
	total = this->faceTrain[0].rows * this->faceTrain[0].cols;
	for( int i = 0; i < this->faceTrain.size(); i++){
		cv :: Mat img = this->faceTrain[i].reshape(1, total);
		cv :: Mat tmp = pca.project(img);
		cv::normalize(tmp, tmp);
		this->dataSVM.push_back(tmp);
	 }
}

void FaceRecognition :: trainSVM(cv :: Mat label, std :: string model){
	clock_t tStart = clock();
	std :: cout << "start training SVM!" << std :: endl;
	cv :: Mat mat (this->faceTrain.size(), this->dataSVM[0].cols * this->dataSVM[0].rows, CV_32FC1);
	int idx = 0;
	for(int i = 0; i < this->dataSVM.size(); i++) {
		cv :: Mat X = mat.row(i);
		dataSVM[idx++].reshape(1, 1).row(0).convertTo(X, CV_32FC1);
	}

	this->svm->train(mat, cv :: ml :: ROW_SAMPLE, label);
	std :: cout << "Saving model!" << std :: endl;
	this->svm->save(model);
	std :: cout << "time: " <<(double)(clock() - tStart)/CLOCKS_PER_SEC << std :: endl;

}

void FaceRecognition:: loadModel(std :: string model){
	this->svm = cv::ml::SVM::load<cv::ml::SVM>(model);
}
void FaceRecognition :: prediction(std :: vector <cv :: Mat> dataTest, cv :: Mat lab){
	int total = dataTest[0].rows * dataTest[0].cols;
	float err = 0;
	for(int idx = 0; idx < dataTest.size(); idx++){
		cv :: Mat img = dataTest[idx].reshape(1, total);
		cv :: Mat tmp = pca.project(img);
		cv::normalize(tmp, tmp);
		cv :: Mat tmpT = tmp.t();
		float pre = svm->predict(tmpT);
		if(lab.at<int>(0, idx) != pre){
			std :: cout << this->namePeoples[lab.at<int>(0, idx)] << " || "<< this->namePeoples[pre];
			std :: cout << "     --> Not Match." << std :: endl;
			err++;
		}
		if(lab.at<int>(0, idx) == pre){
			std :: cout << this->namePeoples[lab.at<int>(0, idx)] << " || "<< this->namePeoples[pre];
			std :: cout << "  --> Match." << std :: endl;
		}
	}
	std :: cout << "Total Image test: " << dataTest.size() << std :: endl;
	std :: cout << "Image Error: " << err << std :: endl;
	std :: cout << "Error rate: " << err / (1.0 * dataTest.size()) << std :: endl;
}

void FaceRecognition :: predictFAR(){
	std :: cout << "predict negative image!" << std :: endl;
	cv :: Mat lb = cv ::Mat::ones(1, this->faceTestFAR.size(), CV_32S);
	lb = lb * MAXPEOPLE;
	this->prediction(this->faceTestFAR, lb);
}

void FaceRecognition :: predictFRR(){
	std :: cout << "predict positive image!" << std:: endl;
	cv :: Mat lb = faceRecognitionCreateLabelTestFRR();
	this->prediction(this->faceTestFRR, lb);
}

cv :: Mat FaceRecognition :: faceRecognitionCreateLabelTrain(){
	cv :: Mat lblTrn(this->faceTrain.size() / MAX_IMAGE_TRAIN, MAX_IMAGE_TRAIN , CV_32S);
	int idx = 0;
	for(int i = 0; i < lblTrn.rows; i++){
		for(int j = 0; j < lblTrn.cols; j++){
			lblTrn.at<int>(i, j) = idx;
		}
		if(i < MAXPEOPLE)
			idx++;
	}
	lblTrn = lblTrn.reshape(1, 1);
	lblTrn = lblTrn.t();
	return lblTrn;
}
cv :: Mat FaceRecognition :: faceRecognitionCreateLabelTestFRR(){
	cv :: Mat lblTst(this->faceTestFRR.size() / (TOTAl_IMAGE - MAX_IMAGE_TRAIN),(TOTAl_IMAGE - MAX_IMAGE_TRAIN) , CV_32S);
	int idx = 0;
	for(int i = 0; i < lblTst.rows; i++){
		for(int j = 0; j < lblTst.cols; j++){
			lblTst.at<int>(i, j) = idx;
		}
		if(i < MAXPEOPLE)
			idx++;
	}
	lblTst = lblTst.reshape(1, 1);
	lblTst = lblTst.t();
	return lblTst;
}
void FaceRecognition :: loadNamePeoples(){
	std :: ifstream nameFileout;
	nameFileout.open("./Sysset_word.txt");
	if(nameFileout.is_open() == false){
		std :: cout << "check file Sysnet word" << std :: endl;
		exit(0);
	}

	std :: string line;
	while(std::getline(nameFileout, line)){
		this->namePeoples.push_back(line);
		line = "";
	}
	nameFileout.close();
}
