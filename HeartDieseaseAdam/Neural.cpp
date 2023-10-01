#include <iostream>
#include <vector>
#include <iomanip>      
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <string>
#include "Neural.h"
using namespace std;
Neural::Neural(int outlayer_c, int secondlayer_c, int firstlayer_c, int zerolayer_c, double learnrate_c) {

	outlayer = outlayer_c;
	secondlayer = secondlayer_c;
	firstlayer = firstlayer_c;
	zerolayer = zerolayer_c;
	learnrate = learnrate_c;
	input_layer.resize(zerolayer);
	first_layer.resize(firstlayer);
	miss_first.resize(firstlayer);
	second_layer.resize(secondlayer);
	miss_second.resize(secondlayer);
	out_layer.resize(outlayer);
	miss_out.resize(outlayer);

	eta = 0.01;
	beta1 = 0.9;
	beta2 = 0.999;
	epsilon = 1e-8;


	weights_0_1.resize(firstlayer, vector <double>(zerolayer));
	weights_1_2.resize(secondlayer, vector <double>(firstlayer));
	weights_2_out.resize(outlayer, vector <double>(secondlayer));
	mdelta_0_1.resize(firstlayer, vector <double>(zerolayer));
	mdelta_1_2.resize(secondlayer, vector <double>(firstlayer));
	mdelta_2_out.resize(outlayer, vector <double>(secondlayer));
	vdelta_0_1.resize(firstlayer, vector <double>(zerolayer));
	vdelta_1_2.resize(secondlayer, vector <double>(firstlayer));
	vdelta_2_out.resize(outlayer, vector <double>(secondlayer));

	for (int i = 0; i < weights_0_1.size(); i++) {
		for (int j = 0; j < weights_0_1[0].size(); j++) {
			weights_0_1[i][j] = (1 + rand() % 200) / 100. - 1;
			mdelta_0_1[i][j] = 0;
			vdelta_0_1[i][j] = 0;
		}
	}
	for (int i = 0; i < weights_1_2.size(); i++) {
		for (int j = 0; j < weights_1_2[0].size(); j++) {
			weights_1_2[i][j] = (1 + rand() % 200) / 100. - 1;
			mdelta_1_2[i][j] = 0;
			vdelta_1_2[i][j] = 0;
		}
	}
	for (int i = 0; i < weights_2_out.size(); i++) {
		for (int j = 0; j < weights_2_out[0].size(); j++) {
			weights_2_out[i][j] = (1 + rand() % 200) / 100. - 1;
			mdelta_2_out[i][j] = 0;
			vdelta_2_out[i][j] = 0;
		}
	}
}
vector<double> Neural::matrix_vector(vector<vector<double> > matrix, vector<double> Vector) {
	vector <double> answer(matrix.size());	// ������������ ������� �� ������ - ������� ������� ��� ������ � �������������
	double sum = 0;							// �������� ���������
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[0].size(); j++) {
			sum = sum + matrix[i][j] * Vector[j];
		}
		answer[i] = sum;
		sum = 0;
	}
	return answer;
}
vector <double> Neural::activation(vector<double> Vector) {	// ��������� ������� ��������� �������, ��� ���������� ��� ����, 
	for (int i = 0; i < Vector.size(); i++) {		// ����� ���������� ��������� �������� �������� � �������� �� 0 �� 1 
		Vector[i] = 1 / (1 + exp(-Vector[i]));		// ��� ������� ������������ ������� ��������
	}
	return Vector;
}

vector<double> Neural::predict(vector<double> input) {		// ������ ��������� - �������� ������������� ��������
	input_layer = input;							// � ��������, ��� ������ ��������� - ��� ���������������� ������������ ����� ������ �� �������� ������� �������
	input_layer = activation(input_layer);			// �������� ���������� � ��������� �������� ����
	first_layer = matrix_vector(weights_0_1, input_layer);	// �������� ������� ���� ������������ ������������� ����� ������ �� �������� �������� ����
	first_layer = activation(first_layer);					// ����� ������������ ���������� ��������� ��������� 
	second_layer = matrix_vector(weights_1_2, first_layer);	// ����� ���� �������� ����������� ������ �� ��������� ����
	second_layer = activation(second_layer);
	out_layer = matrix_vector(weights_2_out, second_layer);
	out_layer = activation(out_layer);
	return out_layer;
}
void Neural::Adam(double miss, int t) {				// �������, �������� ���� ��������� �����			
	miss_out[0] = miss * out_layer[0] * (1 - out_layer[0]);	// � ��������� ������ ���������� �������� ��������� ��������������� ������
	

	
	for (int j = 0; j < secondlayer; j++) {					// ������� ������� �� ������ ������� ������, ������� ������ ����� ������������
		miss_second[j] = miss_out[0] * second_layer[j] * (1 - second_layer[j]) * weights_2_out[0][j];
	}														// 1) ������� ����������� ������ ������� ����
	for (int i = 0; i < firstlayer; i++) {
		miss_first[i] = 0;
		for (int j = 0; j < secondlayer; j++) {
			miss_first[i] = miss_first[i] + miss_second[j] * weights_1_2[j][i] * first_layer[i] * (1 - first_layer[i]);
		}
	}
	for (int i = 0; i < outlayer; i++) {
		for (int j = 0; j < secondlayer; j++) {
			//cout << "___________________________________________\n";
			//cout << "BREATHE";
			m_dw_corr = m_dw_corr_compute(&mdelta_2_out[i][j], miss_out[i] * second_layer[j], t + 1);
			v_dw_corr = v_dw_corr_compute(&vdelta_2_out[i][j], miss_out[i] * second_layer[j], t + 1);
			weights_2_out[i][j] = weights_2_out[i][j] - learnrate * m_dw_corr / (sqrt(v_dw_corr) + epsilon);
			//weights_2_out[i][j] = weights_2_out[i][j] - miss_out[i] * second_layer[j] * learnrate;
		}
	}
	for (int i = 0; i < secondlayer; i++) {
		for (int j = 0; j < firstlayer; j++) {
			m_dw_corr = m_dw_corr_compute(&mdelta_1_2[i][j] ,miss_second[i] * first_layer[j], t + 1);
			v_dw_corr = v_dw_corr_compute(&vdelta_1_2[i][j], miss_second[i] * first_layer[j], t + 1);
			weights_1_2[i][j] = weights_1_2[i][j] - learnrate * m_dw_corr / (sqrt(v_dw_corr) + epsilon);
			//weights_1_2[i][j] = weights_1_2[i][j] - miss_second[i] * first_layer[j] * learnrate;
		}
	}
	for (int i = 0; i < firstlayer; i++) {
		for (int j = 0; j < zerolayer; j++) {
			m_dw_corr = m_dw_corr_compute(&mdelta_0_1[i][j], miss_first[i] * input_layer[j], t + 1);
			v_dw_corr = v_dw_corr_compute(&vdelta_0_1[i][j],miss_first[i] * input_layer[j], t + 1);
			weights_0_1[i][j] = weights_0_1[i][j] - learnrate * m_dw_corr / (sqrt(v_dw_corr) + epsilon);
			//weights_0_1[i][j] = weights_0_1[i][j] - miss_first[i] * input_layer[j] * learnrate;
		}
	}
	/*
	cout << "M_DW_CORR: " << m_dw_corr << endl;
	cout << "V_DW_CORR: " << v_dw_corr << endl;
	cout << "M_DW: " << m_dw << endl;
	cout << "V_DW: " << v_dw << endl;
	cout << "ACTUAL DELTA: " << -learnrate * m_dw_corr / (sqrt(v_dw_corr) + epsilon) << endl;
	*/
}

void Neural::batch_update() {
	for (int i = 0; i < weights_0_1.size(); i++) {
		for (int j = 0; j < weights_0_1[0].size(); j++) {
			mdelta_0_1[i][j] = 0;
			vdelta_0_1[i][j] = 0;
		}
	}
	for (int i = 0; i < weights_1_2.size(); i++) {
		for (int j = 0; j < weights_1_2[0].size(); j++) {
			mdelta_1_2[i][j] = 0;
			vdelta_1_2[i][j] = 0;
		}
	}
	for (int i = 0; i < weights_2_out.size(); i++) {
		for (int j = 0; j < weights_2_out[0].size(); j++) {
			mdelta_2_out[i][j] = 0;
			vdelta_2_out[i][j] = 0;
		}
	}
}


double Neural::v_dw_corr_compute(double* v_dw, double dw, int t) {
	*v_dw = this->beta2 * *v_dw + (1 - this->beta2) * (pow(dw, 2));
	return  *v_dw / (1 - pow(this->beta2, t));
}

double Neural::m_dw_corr_compute(double* m_dw, double dw, int t) {
	*m_dw = this->beta1 * *m_dw + (1 - this->beta1) * dw;
	return *m_dw / (1 - pow(this->beta1, t));
}



Neural::~Neural() {

}