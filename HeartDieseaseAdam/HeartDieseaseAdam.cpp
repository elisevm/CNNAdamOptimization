/*
В данном коде представлена реализация нейронной сети (многослойного перцептрона) обучение которой построено на базе алгоритма обратного распространения ошибки.
Сама нейросеть описана с помощью класса Neural. Применение её представлено на примере данных о результатах диагностики порока сердца.
Данные были взяты с форума по машинному обучениию Kaggle. Ссылка - https://www.kaggle.com/johnsmith88/heart-disease-dataset.
Нейросеть способна предсказать порок сердца с точностью в 84% (видно из консольного вывода). Это далеко не предел для этого набора данных,
однако такая точность свидетельствует о том, что нейросеть действительно обучена и способна использоваться по назначению.
*/
#include <iostream>
#include <vector>
#include <iomanip>      
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <string>
#include "HeartDieseaseAdam.h"
#include "Neural.h"
using namespace std;

vector <vector <double> > LoadHeartData() {	// Это функция, которая представляет csv файл в виде двумерного вектора
	ifstream ip("heart.csv");

	if (!ip.is_open()) std::cout << "ERROR: File Open" << '\n';

	string age;
	string sex;
	string cp;
	string trestbps;
	string chol;
	string fbs;
	string restecg;
	string thalach;
	string exang;
	string oldpeak;
	string slope;
	string ca;
	string thal;
	string target;
	vector <vector <double> > data;
	int rows_size = 0;
	while (ip.good()) {
		rows_size++;
		data.resize(rows_size, vector <double>(14));
		getline(ip, age, ',');
		data[rows_size - 1][0] = atof(age.c_str());
		getline(ip, sex, ',');
		data[rows_size - 1][1] = atof(sex.c_str());
		getline(ip, cp, ',');
		data[rows_size - 1][2] = atof(cp.c_str());
		getline(ip, trestbps, ',');
		data[rows_size - 1][3] = atof(trestbps.c_str());
		getline(ip, chol, ',');
		data[rows_size - 1][4] = atof(chol.c_str());
		getline(ip, fbs, ',');
		data[rows_size - 1][5] = atof(fbs.c_str());
		getline(ip, restecg, ',');
		data[rows_size - 1][6] = atof(restecg.c_str());
		getline(ip, thalach, ',');
		data[rows_size - 1][7] = atof(thalach.c_str());
		getline(ip, exang, ',');
		data[rows_size - 1][8] = atof(exang.c_str());
		getline(ip, oldpeak, ',');
		data[rows_size - 1][9] = atof(oldpeak.c_str());
		getline(ip, slope, ',');
		data[rows_size - 1][10] = atof(slope.c_str());
		getline(ip, ca, ',');
		data[rows_size - 1][11] = atof(ca.c_str());
		getline(ip, thal, ',');
		data[rows_size - 1][12] = atof(thal.c_str());
		getline(ip, target, '\n');
		data[rows_size - 1][13] = atof(target.c_str());

	}

	ip.close();
	return data;
}
int main() {
	int predict_range = 1;
	int first_layer_size = 13;
	int second_layer_size = 4;
	int zero_layer_size = 13;
	double current_learn_rate = 0.007;


	Neural neural = Neural(predict_range, second_layer_size, first_layer_size, zero_layer_size, current_learn_rate);	// Объявим объект класса и укажем конструктор для него

	vector <vector <double> > data = LoadHeartData();	// Загрузим данны из csv файла в вектор
	// Отделим анализы от диагноза:
	vector <vector <double> > X;
	X.resize(data.size(), vector<double>(data[0].size() - 1));
	vector <vector <double> > y;
	y.resize(data.size(), vector<double>(1));

	// В данном цикле мы разделим исходые данные на признаки X и ожидания y
	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < data[0].size() - 1; j++) {
			X[i][j] = data[i][j];
		}
		y[i][0] = data[i][data[0].size() - 1];
	}
	// Разделим X и y на обучающие и тестовые данные:
	vector <vector <double> > X_train;
	X_train.resize(int(X.size() * 0.8), vector<double>(X[0].size()));
	vector <vector <double> > X_test;
	X_test.resize(X.size() - int(X.size() * 0.8), vector<double>(X[0].size()));

	vector <vector <double> > y_train;
	y_train.resize(int(y.size() * 0.8), vector<double>(y[0].size()));
	vector <vector <double> > y_test;
	y_test.resize(y.size() - int(y.size() * 0.8), vector<double>(y[0].size()));

	int k = 0;	// В цикле так же разделим данные на тестовые и тренировочные




	for (int i = 0; i < data.size(); i++) {	// Это нужно для того, чтобы справедливо оценить точность обученной сети
		if (i < int(X.size() * 0.8)) {		// Тестироваться сеть будет на тех данных, которые ей не встречались при обучении
			X_train[i] = X[i];
			y_train[i] = y[i];
		}
		else {
			k = int(i - X.size() * 0.8);
			X_test[k] = X[k];
			y_test[k] = y[k];
		}
	}
	vector <double> target;		// Объявим вспомогателньые вектора
	target.resize(predict_range);
	vector<double> person;
	person.resize(zero_layer_size);
	vector <double> predict = neural.predict(person);
	double miss = 0;			// miss - значение промаха, это то на сколько прогноз нейросети отличается от наших ожиданий
	double count = 0;			// count - количество раз, когда модель угадала диагноз 
	for (int i = 0; i < X_test.size(); i++) {
		person = X_test[i];
		target = y_test[i];
		predict = neural.predict(person);
		if (int(predict[0] + 0.5) == target[0]) {
			count++;
		}
	}
	double accuracy = count / X_test.size();	// Посчитаем точность, поделив count на количество рассмотренных в цикле пациентов
	count = 0;
	cout << "Accuracy before learning: " << accuracy << endl;
	// Теперь обучим нейросеть на тренировочных данных
	for (int j = 0; j < 5; j++) {
		//neural.batch_update();
		for (int i = 0; i < X_train.size(); i++) {
			person = X_train[i];
			target = y_train[i];
			predict = neural.predict(person);
			miss = predict[0] - target[0];
			//if (i%20==0){
			//	neural.batch_update();
			//}
			neural.Adam(miss, i);
		}
	}
	for (int i = 0; i < X_test.size(); i++) {
		person = X_test[i];
		target = y_test[i];
		predict = neural.predict(person);	// Теперь произведём валидацию сети - посчитаем точность на ранее неизвестных сети данных
		miss = target[0] - predict[0];
		if (int(predict[0] + 0.5) == target[0]) {
			count++;
		}
	}
	accuracy = count / X_test.size();		// Вычисляем конечную точность
	cout << "Final accuracy:           " << accuracy << endl;
	cout << "Test predictions: ";
	//cout << neural.m_dw_corr_compute(-0.5, 1);
	//cout << neural.predict(X[55])[0] << " - " << y[55][0] << endl;
	//cout << neural.predict(X[603])[0] << " - " << y[603][0] << endl;
	return 0;
	system("pause");
}
