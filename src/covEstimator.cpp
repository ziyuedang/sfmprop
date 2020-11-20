#include "CovEstimator.h"
#include <cmath>
using namespace std;
using namespace cv;
using namespace openMVG::features;
#define PI		3.14159265f

Image<float> CovEstimator::getImageFromPyramid(int octv, int intvls) {
	// Retrieve the dog at octv, intvls [Tested, works]
	return detectPyr[octv].slices[intvls];
}

MatCv CovEstimator::getCovAt(float x, float y, float scale, int oct, int slice) {

	// Retrieve the octave and interval the feature was detected at
	int row = 0, col = 0;
	float subintv = 0.0f;

	// scale calculation: scl = sigma * 2 ^ ((octv*intlvs + inv)/intvls) * 1/ 2;
	// subintv is the refined interval(slice) index
	subintv = (log2(scale/SIFT_SIGMA) - oct) * intervals;

	// location calculation: feat->x = ( c + xc ) * pow( 2.0, octv );
	// Edit: original was octv - 1
	col = cvRound(x / pow(2.0, oct));
	row = cvRound(y / pow(2.0, oct));
	Image<float> img = getImageFromPyramid(oct, slice);

	// determine hessan at that point and calculate and scale covariance matrix
	MatCv H = hessian(img, row, col);
	invert(H, cov, CV_SVD_SYM);

	// Hessian is estimated at particular octave and interval, thus scaling needed, which
	// adapts for the octave and subinterval
	cov.convertTo(cov, -1, pow(2.0f, (oct + subintv / intervals) ) );
	MatCv vt;
	SVD::compute(cov, evals, evecs, vt);
	ev1 = evals.at<float>(0, 0);
	ev2 = evals.at<float>(1, 0);
	if (ev1 < 0 && ev2 < 0) {
		ev1 = -ev1;
		ev2 = -ev2;
	}
	if (ev1 < ev2) {
		float tmp = ev1;
		ev1 = ev2;
		ev2 = tmp;
	}
	if (ev1 <= 0 || ev2 <= 0) {
		cout << "COV Eigenvalue of Hessian is negative or zero(!)" << endl;
	}

	return cov;
}


MatCv CovEstimator::hessian(Image<float> dog, int row, int col) {
/* dog is the detected dog image, not the entire pyramid */
	int r, c;
	float v, dxx = 0, dyy = 0, dxy = 0;
	float w[3][3] = { 0.0449f,    0.1221f,    0.0449f,
		0.1221f,    0.3319f,    0.1221f,
		0.0449f,    0.1221f,    0.0449f };
	for(int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++) {
			r = row + j - 1;
			c = col + i - 1;
			v = dog(r, c);
			dxx += w[i][j] * (dog(r, c + 1) + dog(r, c - 1) - 2 * v);
			dyy += w[i][j] * (dog(r + 1, c) + dog(r - 1, c) - 2 * v);
			dxy += w[i][j] * ((dog(r + 1, c + 1) -
				dog(r + 1, c - 1) -
				dog(r - 1, c + 1) +
				dog(r - 1, c - 1)) / 4.0f);
		}
	MatCv H(2, 2, CV_32FC1);
	H.at<float>(0, 0) = -dxx;
	H.at<float>(0, 1) = -dxy;
	H.at<float>(1, 0) = -dxy;
	H.at<float>(1, 1) = -dyy;

	return H;
}