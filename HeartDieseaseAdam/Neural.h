#pragma once
#ifndef __NEURAL_INCLUDED__
#define __NEURAL_INCLUDED__
#include <vector>
using namespace std;
class Neural {
private:
	int outlayer, secondlayer, firstlayer, zerolayer;
	vector<double> input_layer, first_layer, miss_first, second_layer, miss_second, out_layer, miss_out;
	vector<vector<double> > weights_0_1, weights_1_2, miss_2_out, weights_2_out;
	vector<vector<double> > mdelta_0_1, mdelta_1_2, mdelta_2_out, vdelta_0_1, vdelta_1_2, vdelta_2_out;
public:
	Neural(int outlayer_c, int secondlayer_c, int firstlayer_c, int zerolayer_c, double learnrate_c);
	~Neural();
	vector <double> matrix_vector(vector<vector<double> > matrix, vector<double> Vector);
	vector <double> activation(vector<double> Vector);
	vector <double> predict(vector<double> input);
	
	double m_dw_corr_compute(double* m_dw, double dw, int t);
	double v_dw_corr_compute(double* v_dw, double dw, int t);

	void batch_update();
	void Adam(double miss, int t);
	double learnrate, beta1, beta2, eta, epsilon, m_dw_corr, v_dw_corr;
};
#endif
