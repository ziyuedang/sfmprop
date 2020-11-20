#define _USE_MATH_DEFINES
//c/c++
#include <iostream>
#include <string>
#include <cmath>

//opencv
#include <opencv2/imgcodecs.hpp>

//openMVG
#include "openMVG/image/image_container.hpp"
#include "openMVG/image/image_io.hpp"
#include "openMVG/features/feature.hpp"
#include "openMVG/features/sift/SIFT_Anatomy_Image_Describer_io.hpp"
#include "openMVG/features/sift/sift_KeypointExtractor.hpp" 
#include "openMVG/features/akaze/image_describer_akaze_io.hpp"

// Zeisl's method
#include "covEstimator.h"

//own
#include "covArgEvaluator.h"
#include "fileIO.h"

using namespace std;
using namespace openMVG;
using namespace openMVG::image;
using namespace openMVG::features;
using namespace openMVG::features::sift;
using MatCv = cv::Mat;

// Define a feature and a container of features
using Feature_T = SIOPointFeature;
using Feats_T = vector<Feature_T>;

int main(int argc, char* argv[])
{
	string filename;
	Feats_T vec_feats;

	/*** Parsing input arguments ***/
	covArgEvaluator arg;
	arg.evaluate(argc, argv);

	/*** Loading input images and according keys ***/
	// Loading image
	filename.clear();
	filename.append(arg.imgDir).append(arg.imgFile);
	const string imageFile = filename;

	Image<unsigned char> in;

	int res = ReadImage(imageFile.c_str(), &in);
	/*** Check if image is loaded fine ***/
	if (!res) {
		cout << " Cov Error: Unable to load image from " << filename << "\n";
		exit(1);
	}

	/*** Loading the keypoints ***/
	filename.clear();
	filename.append(arg.imgDir).append(arg.keyFile);
	//if (arg.verbose)
	//	cout << "Loading key points from file " << filename << endl;
	float peak_threshold = std::strtod(arg.peak_thresh, NULL);
	float edge_threshold = std::strtod(arg.edge_thresh, NULL);
	//loadFeatsFromFile(filename, vec_feats);

	/*** Creating image pyramid from openMVG***/
	const int supplementary_images = 3;
	int n_octaves = 6;
	int n_levels = 3;
	HierarchicalGaussianScaleSpace octave_gen(n_octaves, n_levels, GaussianScaleSpaceParams(1.6f, 1.0f, 0.5, supplementary_images));
	Image<float> image(in.GetMat().cast<float>() / 255.0f);
	Octave octave;
	Octave m_dogs;
	octave_gen.SetImage(image);

	// Generate Difference of Gaussian and keypoints
	vector<Keypoint> keypoints;
	vector<Octave> dog_pyr;
	keypoints.reserve(5000);
	cerr << "Octave computation started" << std::endl;
	uint8_t octave_id = 0;
	while (octave_gen.NextOctave(octave))
	{
		cerr << "Computed octave : " << std::to_string(octave_id) << std::endl;
		vector<Keypoint> keys;
		SIFT_KeypointExtractor keypointDetector(peak_threshold / octave_gen.NbSlice(), edge_threshold, 5);
		keypointDetector(octave, keys);
		Sift_DescriptorExtractor descriptorExtractor;
		descriptorExtractor(octave, keys);

		// Concatenante the keypoints
		move(keys.begin(), keys.end(), back_inserter(keypoints));

		// Generate dog
		const int n = octave.slices.size();
		m_dogs.slices.resize(n - 1);
		m_dogs.octave_level = octave.octave_level;
		m_dogs.delta = octave.delta;
		m_dogs.sigmas = octave.sigmas;

		for (int i = 0; i < m_dogs.slices.size(); ++i) {
			const Image<float> &P = octave.slices[i + 1];
			const Image<float>& M = octave.slices[i];
			m_dogs.slices[i] = P - M;
		}
		dog_pyr.push_back(m_dogs);
		++octave_id;
	}
	cout << "# of keypoints detected: " << keypoints.size() << endl;
	// Estimate covariance
	string filename_cov_out = filename + ".cov";
	ofstream& outfile = fileOut::initializeFile(filename_cov_out);
	CovEstimator CovEstimator(dog_pyr, n_octaves, n_levels);
	cout << "Writing covariance to file..." << endl;
	cout << "[";
	for (size_t i = 0; i < keypoints.size(); ++i) {
		const Keypoint & key = keypoints[i];
		MatCv cov = CovEstimator.getCovAt(key.x, key.y, key.sigma, key.o, key.s);
		float percentage = 100 * i / keypoints.size();
		if (percentage <= 100 && (percentage - int(percentage)) ==0)
			cout << "#";
		fileOut::write(outfile, cov, key.x, key.y);
	}
	cout << "]" << endl;
	fileOut::closeFile(outfile);
}
