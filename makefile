CC = g++
CFLAGS  = -g -Wall -std=c++11

LIBS_opencv=-L/usr/local/lib -lopencv_ccalib -lopencv_face -lopencv_xfeatures2d -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_xobjdetect -lopencv_objdetect -lopencv_ml -lopencv_xphoto -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_photo -lopencv_imgproc -lopencv_core

default: face_recog

face_recog:  main.o pca.o 
	$(CC) $(CFLAGS) -o face_recog main.o pca.o $(LIBS_opencv)
main.o:  main.cpp pca.cpp pca.h 
	$(CC) $(CFLAGS) -c main.cpp pca.cpp pca.h $(LIBS_opencv)
pca.o:  pca.cpp pca.h 
	$(CC) $(CFLAGS) -c pca.cpp pca.h $(LIBS_opencv)
clean: 
	$(RM) face_recog *.o *~ main pca.h.gch
