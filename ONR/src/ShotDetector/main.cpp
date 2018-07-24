#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "infix_iterator.h"

#include "frame_buffer.h"
#include "evaluator.h"
#include "thresholder.h"

#include <filesystem>

using namespace cv;
using namespace std;
using namespace std::tr2::sys;


inline string zero_padded_string(int _Val, int _MinLength)
{	// convert int to string with leading zeros
	char _Buf[2 * _MAX_INT_DIG];

	sprintf_s(_Buf, sizeof(_Buf), "%0*d", _MinLength, _Val);
	return (string(_Buf));
}

void resize_prop_rect(InputArray src, OutputArray dst, Size dsize, int interpolation = INTER_LINEAR) {
	Size ssize = src.size();

	double xscale = double(dsize.width) / ssize.width;
	double yscale = double(dsize.height) / ssize.height;

	double scale = min(xscale, yscale);

	resize(src, dst, Size(), scale, scale, interpolation);
}


int main (int argc, const char *argv[])
{ 
	// Maximum image size: video will be scaled to this resolution
	const Size max_image_size(320, 180);

	// Algorithm parameters
	const double threshold = 80.0;
	const double threshold_time = 20.0;
	const double threshold_peak = threshold / 2.0;

	// Parameter parsing 
	if (argc != 3) {
		cout << "Syntax:\n\n";
		cout << "ilsd <input_video> <W_size>\n\n";
		exit(EXIT_FAILURE);
	}

	string video_filename(argv[1]);
	unsigned W = atoi(argv[2]);

	// Open video 
	VideoCapture cap(video_filename);
	if (!cap.isOpened()) {
		std::cerr << "Unable to open video.\n";
		exit(EXIT_FAILURE);
	}

	// Find files prefix
	string input_prefix = video_filename.substr(0, video_filename.rfind('.'));

	string gt_filename = input_prefix + "_gt.txt";
	string comparison_filename = input_prefix + "_comparison.txt";
	string diffs_filename = input_prefix + "_diffs.txt";
	string trans_filename = input_prefix + "_trans.txt";
	string shots_filename = input_prefix + "_shots.txt";

	// Save all frames
	if (false) {
		Mat3b img;
		int i = 0;
		path folder_name = input_prefix;
		create_directory(folder_name);

		int nframes = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
		int nchar = static_cast<int>(to_string(nframes).size());

		while (true) {
			if (i % 1000 == 0)
				cout << "\r" << i << " - " << i + 999;
			cap >> img;
			if (img.empty())
				return 0;
			resize_prop_rect(img, img, max_image_size, INTER_AREA);
			path filename = zero_padded_string(i, nchar) + ".png";
			imwrite(string(folder_name / filename), img);
			i++;
		}
		return 0;
	}

	// Ground truth vector
	transitions trans;

	cout << "Ground truth file (" << gt_filename << ") ";
	ifstream is_gt(gt_filename);
	if (is_gt) {
		cout << "found.\n";

		transitions shots;
		copy(istream_iterator<transition>(is_gt), istream_iterator<transition>(), back_inserter(shots));

		double nframes = cap.get(CAP_PROP_FRAME_COUNT);
		for (int i = 1; i < shots.size(); ++i) {
			trans.emplace_back(shots[i - 1]._end + 0.5, shots[i]._beg - 0.5);
		}
		if (shots.back()._end < nframes - 1)
			trans.emplace_back(shots.back()._end + 0.5, nframes - 1);
	}
	else {
		cout << "not found.\n";
	}

	// Comparison vector
	transitions trans_comparison;

	cout << "Comparison file (" << comparison_filename << ") ";
	ifstream is_comparison(comparison_filename);
	if (is_comparison) {
		cout << "found.\n";

		transitions shots;
		copy(istream_iterator<transition>(is_comparison), istream_iterator<transition>(), back_inserter(shots));

		double nframes = cap.get(CAP_PROP_FRAME_COUNT);
		for (int i = 1; i < shots.size(); ++i) {
			trans_comparison.emplace_back(shots[i - 1]._end + 0.5, shots[i]._beg - 0.5);
		}
		if (shots.back()._end < nframes - 1)
			trans_comparison.emplace_back(shots.back()._end + 0.5, nframes - 1);
	}
	else {
		cout << "not found.\n";
	}

	// Differences vector
	vector<diff> diffs;

	cout << "Differences file (" << diffs_filename << ") ";
	ifstream is_diffs(diffs_filename);
	if (is_diffs) {
		cout << "already present: video won't be analyzed.\n";
		cout << "Reading differences... ";

		copy(istream_iterator<diff>(is_diffs), istream_iterator<diff>(), back_inserter(diffs));

		cout << "done.\n";
	}
	else {
		cout << "not found: video will be analyzed.\n";
		cout << "Analyzing video... \n";

		int index = -1;
		frame_buffer fb(W);

		while (true) {
			index++;
			if (index % 1000 == 0)
				cout << "\r" << index << " - " << index + 999;

			// Limit maximum number of frames to be analyzed
			//if (index == 4000)
			//	break;

			Mat3b img;
			cap >> img;
			if (img.empty())
				break;
			resize_prop_rect(img, img, max_image_size, INTER_AREA);
			fb.push_back(img);

			if (index >= int(W)) {
				if (index - W > 0)
					diffs.push_back(fb.get_diffs(index - W - 0.5));
				diffs.push_back(fb.get_diffs(index - W));
			}

		}
		cout << "\n";
		cout << "done.\n";

		cout << "Writing differences... ";
		ofstream os_diffs(diffs_filename, ofstream::out);
		if (!os_diffs) {
			cerr << "Unable to open differences file for writing.\n";
			exit(EXIT_FAILURE);
		}
		copy(diffs.begin(), diffs.end(), ostream_iterator<diff>(os_diffs, "\n"));
		cout << "done.\n";
	}

	string ev_filename = input_prefix + "_eval.txt";
	ofstream os_ev;
	if (!trans.empty()) {
		os_ev.open(ev_filename);
	}

	vector<vector<int>> out_trans(diffs.size(), vector<int>(2 * W, 0));

	thresholder thr(diffs);
	for (double w=0.5; w<=W; w += 0.5) {
		thr.apply(w, threshold, threshold_time, threshold_peak);

		for (const auto& tran : thr._trans) {
			for (double frame = tran._beg; frame <= tran._end; frame += 0.5) {
				out_trans[int(frame * 2)][int(w * 2 - 1)] = 140;
			}
		}

		if (!trans.empty()) {
			evaluator ev(trans, thr._trans);
			double precision = double(ev._p._tot._num[0]) / (double(ev._p._tot._num[0]) + double(ev._p._tot._num[1]));
			double recall = double(ev._p._tot._num[0]) / (double(ev._p._tot._num[0]) + double(ev._p._tot._num[2]));
			double F1 = 2 * precision*recall / (precision + recall);
			cout << "W      :\t" << w << "\n";
			cout << "Total  :\t" << ev._p._tot << "\t" << F1 << "\n";
			cout << "Abrupt :\t" << ev._p._split[transition::abrupt] << "\n";
			cout << "Gradual:\t" << ev._p._split[transition::gradual] << "\n";
			cout << "\n";

			os_ev << "Transition log for W=" << w << "\n";
			os_ev << "Total:\n";
			ev._p._tot.log_trans(os_ev);
			os_ev << "Abrupt:\n";
			ev._p._split[transition::abrupt].log_trans(os_ev);
			os_ev << "Gradual:\n";
			ev._p._split[transition::gradual].log_trans(os_ev);
			os_ev << "--------------------------------------------------\n";
		}
	}

	if (!trans_comparison.empty()) {
		evaluator ev(trans, trans_comparison);
		double precision = double(ev._p._tot._num[0]) / (double(ev._p._tot._num[0]) + double(ev._p._tot._num[1]));
		double recall = double(ev._p._tot._num[0]) / (double(ev._p._tot._num[0]) + double(ev._p._tot._num[2]));
		double F1 = 2 * precision*recall / (precision + recall);
		cout << "Comparison with " << comparison_filename << ":\n";
		cout << "Total  :\t" << ev._p._tot << "\t" << F1 << "\n";
		cout << "Abrupt :\t" << ev._p._split[transition::abrupt] << "\n";
		cout << "Gradual:\t" << ev._p._split[transition::gradual] << "\n";
		cout << "\n";
	}

	cout << "Writing transitions (" << trans_filename << ")... ";
	ofstream os_trans(trans_filename);
	if (!os_trans) {
		cerr << "Unable to open transitions file for writing.\n";
		exit(EXIT_FAILURE);
	}
	for (const auto& t : out_trans) {
		copy(t.begin(), t.end(), infix_ostream_iterator<int>(os_trans, "\t"));
		os_trans << "\n";
	}
	cout << "done.\n";


	if (true) {
		cout << "Writing shots (" << shots_filename << ")... ";

		double nframes = cap.get(CAP_PROP_FRAME_COUNT);

		transitions shots;
		double beg = 0;
		for (const auto& x : thr._trans) {
			shots.emplace_back(beg, floor(x._beg - 0.5));
			beg = ceil(x._end + 0.5);
		}
		if (beg < nframes - 1)
			shots.emplace_back(beg, nframes - 1);

		ofstream os_shots(shots_filename);
		if (!os_shots) {
			cerr << "Unable to open shots file for writing.\n";
			exit(EXIT_FAILURE);
		}
		copy(shots.begin(), shots.end(), ostream_iterator<transition>(os_shots));

		cout << "done.\n";
	}
}




