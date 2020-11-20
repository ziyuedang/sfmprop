#pragma once
//c++
#include <iostream>

//opencv
#include <opencv2/core/core_c.h>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

//openMVG
#include "openMVG/image/image_container.hpp"
#include "openMVG/image/image_io.hpp"
#include "openMVG/features/feature.hpp"
#include "openMVG/features/sift/SIFT_Anatomy_Image_Describer.hpp"
#include "openMVG/features/sift/sift_KeypointExtractor.hpp"

//covUtils
#include "definitions.h"
using MatCv = cv::Mat;
using namespace openMVG::features;
using namespace openMVG::image;
class CovEstimator {

public:
	CovEstimator(std::vector<Octave> pyr, int octvs, int intvls) {
		detectPyr = pyr;
		octaves = octvs;
		intervals = intvls;
//		MatCv H(2, 2, CV_32FC1);
		MatCv cov(2, 2, CV_32FC1);
		MatCv evals(2, 1, CV_32FC1);
		MatCv evecs(2, 2, CV_32FC1);
	};

	Image<float> getImageFromPyramid(int octv, int intvls);
	MatCv getCovAt(float x, float y, float scale, int oct, int slice);


private:
	/*** Methods ***/
	MatCv hessian(Image<float> img, int r, int c);


	/*** Member variables ***/
	int type;
	int octaves, intervals;
	std::vector<Octave> detectPyr;
//	MatCv H;
	MatCv cov;
	MatCv evals;
	float ev1, ev2;
	MatCv evecs;
};
