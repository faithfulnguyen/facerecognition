#include "pca.h"

#define TRAIN
int main(){

	FaceRecognition obj;
	obj.readDataTrain();
	obj.readDataTestFAR();
	obj.readImageTrain();
	obj.readImageTestFRR();
	obj.readImageTestFAR();
	obj.doPCA();
	cv :: Mat label = obj.faceRecognitionCreateLabelTrain();
#ifdef TRAIN
	obj.trainSVM(label, "./SVM_model.prototxt");
#else
	obj.loadModel("./SVM_model.prototxt");
#endif
	//predict
	obj.predictFAR();
	obj.predictFRR();

}
