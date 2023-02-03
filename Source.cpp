#include <tbb/tbb.h>
#include <tbb/info.h>
#include <omp.h>
#include <iostream>
#include <locale>

#define EXP_NUM 100

using namespace tbb;

template <typename t>
class reduce_par
{
	t** array1, ** array2;
	t sum;
public:
	reduce_par(t** _array1, t** _array2) : sum(0.), array1(_array1), array2(_array2) {}
	// Splitting constructor
	reduce_par(const reduce_par& r, split) : sum(0.), array1(r.array1), array2(r.array2) {}

	void operator() (const blocked_range2d<size_t>& r)
	{
		for (size_t i = r.rows().begin(); i < r.rows().end(); i++)
			for (int j = r.cols().begin(); j < r.cols().end(); j++)
			{
				sum += array1[i][j];
				sum += array2[i][j];
			}
	}
	// Объединение двух задач
	void join(const reduce_par& r) { sum += r.sum; }
};

template <typename t>
class addMatrix
{
	t** array1;
	t** array2;
	t** result;
	int size;
public:
	addMatrix(t** _array1, t** _array2, t** _result) : result(_result), array1(_array1), array2(_array2) {}
	// Splitting constructor
	addMatrix(const addMatrix& r, split) : result(r.result), array1(r.array1), array2(r.array2) {}

	void operator() (const blocked_range2d<size_t>& r)
	{

		for (size_t i = r.rows().begin(); i < r.rows().end(); i++)
			for (int j = r.cols().begin(); j < r.cols().end(); j++)
			{
				result[i][j] = array1[i][j] + array2[i][j];
			}
	}

	void join(const addMatrix& r) {}
};

template <typename t>
class multiplyMatrix
{
	t** array1;
	t** array2;
	t** result;
public:
	multiplyMatrix(t** _array1, t** _array2, t** _result) : result(_result), array1(_array1), array2(_array2) {}
	// Splitting constructor
	multiplyMatrix(const multiplyMatrix& r, split) : result(r.result), array1(r.array1), array2(r.array2) {}

	void operator() (const blocked_range2d<size_t>& r)
	{
		for (size_t i = r.rows().begin(); i < r.rows().end(); i++)
			for (int j = r.cols().begin(); j < r.cols().end(); j++)
				result[i][j] = array1[i][j] * array2[i][j];
	}

	void join(const multiplyMatrix& r) {}
};

template <typename t>
class MinMaxCalc
{
	t** array1;
	t** array2;
public:
	t minValue;
	t maxValue;
	MinMaxCalc(t** _array1, t** _array2) : array1(_array1), array2(_array2), minValue(DBL_MAX), maxValue(DBL_MIN) {}

	MinMaxCalc(const MinMaxCalc& r, split) : array1(r.array1), array2(r.array2), minValue(DBL_MAX), maxValue(DBL_MIN) {}

	void operator()(const blocked_range2d<size_t>& r)
	{
		for (size_t i = r.rows().begin(); i < r.rows().end(); i++)
			for (int j = r.cols().begin(); j < r.cols().end(); j++)
			{
				if (array1[i][j] < minValue)
					minValue = array1[i][j];
				if (array2[i][j] < minValue)
					minValue = array2[i][j];
				if (array1[i][j] > maxValue)
					maxValue = array1[i][j];
				if (array2[i][j] > maxValue)
					maxValue = array2[i][j];
			}
	}

	void join(const MinMaxCalc& r)
	{
		if (r.minValue < minValue) minValue = r.minValue;
		if (r.maxValue > maxValue) maxValue = r.maxValue;
	}
};

// Шаблонная функция заполнения матриц
template <typename t>
double fillMatr(t**& matrix, int& size)
{
	double time_start = omp_get_wtime();

#pragma omp parallel for
	for (int i = 1; i < size; i++) {
		for (int j = 0; j < size; j++) {
			matrix[i][j] = sin(i * 0.3) + cos(j / 0.3);
		}
	}
	return omp_get_wtime() - time_start;
}

// Шаблонная функция сложения матриц omp parallel for
template <typename t>
double addMatrixOmpParallel(t**& matr1, t**& matr2, int &size)
{

	t** result = new t * [size];
	result[0] = new t[size * size];
	for (int i = 0; i < size; i++)
		result[i] = &result[0][i * size];

	double t_start = omp_get_wtime();

#pragma omp parallel for
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
		{
			result[i][j] = matr1[i][j] + matr2[i][j];
		}

	delete[] result[0], delete[] result;
	return omp_get_wtime() - t_start;
}

// Шаблонная функция сложения матриц omp parallel sections
template <typename t>
double addMatrixOmpSections(t**& matr1, t**& matr2, int &size)
{
	t** result = new t * [size];
	result[0] = new t[size * size];
	for (int i = 0; i < size; i++)
		result[i] = &result[0][i * size];

	int n_t = omp_get_max_threads();
	int st = 0;
	int s1 = size / n_t, s2 = size * 2 / n_t,
		s3 = size * 3 / n_t, se = size;
	int i = 0;

	double t_start = omp_get_wtime();

#pragma omp parallel sections private(i)
	{
#pragma omp section
		{
			for (i = st; i < s1; i++)
				for (int j = 0; j < size; j++)
					result[i][j] = matr1[i][j] + matr2[i][j];
		}
#pragma omp section
		{
			if (n_t > 1)
				for (i = s1; i < s2; i++)
					for (int j = 0; j < size; j++)
						result[i][j] = matr1[i][j] + matr2[i][j];
		}
#pragma omp section
		{
			if (n_t > 2)
				for (i = s2; i < s3; i++)
					for (int j = 0; j < size; j++)
						result[i][j] = matr1[i][j] + matr2[i][j];
		}
#pragma omp section
		{
			if (n_t > 3)
				for (i = s3; i < se; i++)
					for (int j = 0; j < size; j++)
						result[i][j] = matr1[i][j] + matr2[i][j];
		}
	}

	delete[] result[0], delete[] result;

	return omp_get_wtime() - t_start;
}

// Шаблонная функция сложения матриц tbb::parallel_for lambda
template <typename t>
double addMatrixTbbLambda(t**& matr1, t**& matr2, int& size)
{

	t** result = new t * [size];
	result[0] = new t[size * size];
	for (int i = 0; i < size; i++)
		result[i] = &result[0][i * size];

	double t_start = omp_get_wtime();

	tbb::parallel_for(tbb::blocked_range2d<int>(0., size, 0., size),
		[&](tbb::blocked_range2d<int> r)
		{
			for (int i = r.rows().begin(); i < r.rows().end(); i++)
				for (int j = r.cols().begin(); j < r.cols().end(); j++)
					result[i][j] = matr1[i][j] + matr2[i][j];
		});

	delete[] result[0], delete[] result;
	return omp_get_wtime() - t_start;
}

// Шаблонная функция сложения матриц tbb::parallel_reduce class
template <typename t>
double addMatrixTbbClass(t**& matr1, t**& matr2, int& size)
{
	t** result = new t * [size];
	result[0] = new t[size * size];
	for (int i = 0; i < size; i++)
		result[i] = &result[0][i * size];

	double t_start = omp_get_wtime();

	addMatrix<t> r(matr1, matr2, result);
	parallel_reduce(blocked_range2d<size_t>(0., size, 0., size), r);

	delete[] result[0], delete[] result;
	return omp_get_wtime() - t_start;
}

// Шаблонная функция перемножения матриц omp parallel for
template <typename t>
double multiplyMatrixOmpParallel(t**& matr1, t**& matr2, int& size)
{
	t** result = new t * [size];
	result[0] = new t[size * size];
#pragma omp parallel for
	for (int i = 0; i < size; i++)
		result[i] = &result[0][i * size];

	double t_start = omp_get_wtime();

#pragma omp parallel for
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			result[i][j] = matr1[i][j] * matr2[i][j];
	delete[] result[0], delete[] result;

	return omp_get_wtime() - t_start;
}

// Шаблонная функция перемножения матриц omp parallel sections
template <typename t>
double multiplyMatrixOmpSections(t**& matr1, t**& matr2, int& size)
{

	int n_t = omp_get_max_threads();
	int st = 0;

	int s1 = size / n_t, s2 = size * 2 / n_t,
		s3 = size * 3 / n_t, se = size;
	int i = 0;

	t** result = new t * [size];
	result[0] = new t[size * size];
	for (int i = 0; i < size; i++)
		result[i] = &result[0][i * size];

	double t_start = omp_get_wtime();

#pragma omp parallel sections private(i)
	{
#pragma omp section
		{
			for (i = st; i < s1; i++)
				for (int j = 0; j < size; j++)
				{
					result[i][j] = matr1[i][j] * matr2[i][j];
				}
		}
#pragma omp section
		{
			if (n_t > 1)
				for (i = s1; i < s2; i++)
					for (int j = 0; j < size; j++)
						result[i][j] = matr1[i][j] * matr2[i][j];
		}
#pragma omp section
		{
			if (n_t > 2)
				for (int j = 0; j < size; j++)
					for (int j = 0; j < size; j++)
						result[i][j] = matr1[i][j] * matr2[i][j];
		}
#pragma omp section
		{
			if (n_t > 3)
				for (int j = 0; j < size; j++)
					for (int j = 0; j < size; j++)
						result[i][j] = matr1[i][j] * matr2[i][j];
		}
	}

	delete[] result[0], delete[] result;
	return omp_get_wtime() - t_start;
}

// Шаблонная функция перемножения матриц tbb::parallel_for lambda
template <typename t>
double multiplyMatrixTbbLambda(t**& matr1, t**& matr2, int& size)
{

	t** result = new t * [size];
	result[0] = new t[size * size];
	for (int i = 0; i < size; i++)
		result[i] = &result[0][i * size];

	double t_start = omp_get_wtime();

	tbb::parallel_for(tbb::blocked_range2d<int>(0., size, 0., size),
		[&](tbb::blocked_range2d<int> r)
		{
			for (int i = r.rows().begin(); i < r.rows().end(); i++)
				for (int j = r.cols().begin(); j < r.cols().end(); j++)
					result[i][j] = matr1[i][j] * matr2[i][j];
		});
	delete[] result[0], delete[] result;
	return omp_get_wtime() - t_start;
}

// Шаблонная функция перемножения матриц tbb::parallel_reduce class
template <typename t>
double multiplyMatrixTbbClass(t**& matr1, t**& matr2, int& size)
{
	t** result = new t * [size];
	result[0] = new t[size * size];
	for (int i = 0; i < size; i++)
		result[i] = &result[0][i * size];

	double t_start = omp_get_wtime();

	multiplyMatrix<t> r(matr1, matr2, result);
	parallel_reduce(blocked_range2d<size_t>(0., size, 0., size), r);

	delete[] result[0], delete[] result;
	return omp_get_wtime() - t_start;
}

// Шаблонная функция вычислением суммы всех элементов матриц omp parallel for
template <typename t>
double summMatrixOmpParallel(t**& matr1, t**& matr2, int& size)
{
	double t_start = omp_get_wtime();
	t temp = 0;
#pragma omp parallel for reduction (+:temp)
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			temp += matr1[i][j];
			temp += matr2[i][j];

	return omp_get_wtime() - t_start;
}

// Шаблонная функция вычислением суммы всех элементов матриц omp parallel sections
template <typename t>
double summMatrixOmpSections(t**& matr1, t**& matr2, int& size)
{
	int n_t = omp_get_max_threads();
	int st = 0;
	int s1 = size / n_t, s2 = size * 2 / n_t,
		s3 = size * 3 / n_t, se = size;
	int i = 0;
	t temp = 0;

	double t_start = omp_get_wtime();

#pragma omp parallel sections private(i) reduction(+:temp)
	{
#pragma omp section
		{
			for (i = st; i < s1; i++)
				for (int j = 0; j < size; j++)
				{
					temp += matr1[i][j];
					temp += matr2[i][j];
				}
		}
#pragma omp section
		{
			if (n_t > 1)
				for (i = s1; i < s2; i++)
					for (int j = 0; j < size; j++)
					{
						temp += matr1[i][j];
						temp += matr2[i][j];
					}
		}
#pragma omp section
		{
			if (n_t > 2)
				for (i = s2; i < s3; i++)
					for (int j = 0; j < size; j++)
					{
						temp += matr1[i][j];
						temp += matr2[i][j];
					}
		}
#pragma omp section
		{
			if (n_t > 3)
				for (i = s3; i < se; i++)
					for (int j = 0; j < size; j++)
					{
						temp += matr1[i][j];
						temp += matr2[i][j];
					}
		}
	}

	return omp_get_wtime() - t_start;
}

// Шаблонная функция вычислением суммы всех элементов матриц tbb::parallel_for lambda
template <typename t>
double summMatrixTbbLambda(t**& matr1, t**& matr2, int& size)
{
	double t_start = omp_get_wtime();
	auto total = parallel_reduce(
		// const Range &range
		blocked_range2d<size_t>(0., size, 0., size),
		// default num
		0.,
		// const Lambda []
		[&](blocked_range2d<size_t> r, t running_total)
		{
			for (int i = r.rows().begin(); i < r.rows().end(); i++)
			for (int j = r.cols().begin(); j < r.cols().end(); j++)
			{
				running_total += matr1[i][j];
				running_total += matr2[i][j];
			}
	return running_total;
		},
		std::plus<double>());

	return omp_get_wtime() - t_start;
}

// Шаблонная функция вычислением суммы всех элементов матриц tbb::parallel_reduce class
template <typename t>
double summMatrixTbbClass(t**& matr1, t**& matr2, int& size)
{
	double t_start = omp_get_wtime();

	reduce_par<t> r(matr1, matr2);
	parallel_reduce(blocked_range2d<size_t>(0., size, 0., size), r);

	return omp_get_wtime() - t_start;
}

// Шаблонная функция вычисления значения суммы максимального и минимального элемента среди всех элементов двух исходных матриц omp parallel for
template <typename t>
double minmaxOmpParallel(t**& matr1, t**& matr2, int& size)
{
	double t_start = omp_get_wtime();
	
	t minValue = DBL_MAX, maxValue = DBL_MIN;

#pragma omp parallel for
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
		{
			if (matr1[i][j] < minValue)
				minValue = matr1[i][j];
			if (matr2[i][j] < minValue)
				minValue = matr2[i][j];
			if (matr1[i][j] > maxValue)
				maxValue = matr1[i][j];
			if (matr2[i][j] > maxValue)
				maxValue = matr2[i][j];
		}

	return omp_get_wtime() - t_start;
}

// Шаблонная функция вычисления значения суммы максимального и минимального элемента среди всех элементов двух исходных матриц omp parallel sections
template <typename t>
double minmaxOmpSections(t**& matr1, t**& matr2, int& size)
{
	int n_t = omp_get_max_threads();
	int st = 0;
	int s1 = size / n_t, s2 = size * 2 / n_t,
		s3 = size * 3 / n_t, se = size;
	int i = 0;

	t minValue = DBL_MAX, maxValue = DBL_MIN;

	double t_start = omp_get_wtime();

#pragma omp parallel sections private(i)
	{
#pragma omp section
		{
			for (i = st; i < s1; i++)
				for (int j = 0; j < size; j++)
				{
					if (matr1[i][j] < minValue)
						minValue = matr1[i][j];
					if (matr2[i][j] < minValue)
						minValue = matr2[i][j];
					if (matr1[i][j] > maxValue)
						maxValue = matr1[i][j];
					if (matr2[i][j] > maxValue)
						maxValue = matr2[i][j];
				}
		}
#pragma omp section
		{
			if (n_t > 1)
				for (i = s1; i < s2; i++)
					for (int j = 0; j < size; j++)
					{
						if (matr1[i][j] < minValue)
							minValue = matr1[i][j];
						if (matr2[i][j] < minValue)
							minValue = matr2[i][j];
						if (matr1[i][j] > maxValue)
							maxValue = matr1[i][j];
						if (matr2[i][j] > maxValue)
							maxValue = matr2[i][j];
					}
		}
#pragma omp section
		{
			if (n_t > 2)
				for (i = s2; i < s3; i++)
					for (int j = 0; j < size; j++)
					{
						if (matr1[i][j] < minValue)
							minValue = matr1[i][j];
						if (matr2[i][j] < minValue)
							minValue = matr2[i][j];
						if (matr1[i][j] > maxValue)
							maxValue = matr1[i][j];
						if (matr2[i][j] > maxValue)
							maxValue = matr2[i][j];
					}
		}
#pragma omp section
		{
			if (n_t > 3)
				for (i = s3; i < se; i++)
					for (int j = 0; j < size; j++)
					{
						if (matr1[i][j] < minValue)
							minValue = matr1[i][j];
						if (matr2[i][j] < minValue)
							minValue = matr2[i][j];
						if (matr1[i][j] > maxValue)
							maxValue = matr1[i][j];
						if (matr2[i][j] > maxValue)
							maxValue = matr2[i][j];
					}
		}
	}

	return omp_get_wtime() - t_start;
}

// Шаблонная функция вычисления значения суммы максимального и минимального элемента среди всех элементов двух исходных матриц tbb::parallel_for lambda
template <typename t>
double minmaxTbbLambda(t**& matr1, t**& matr2, int& size)
{
	t minValue = DBL_MAX, maxValue = DBL_MIN;

	double t_start = omp_get_wtime();

	auto total = parallel_reduce(
		// const Range &range
		blocked_range2d<size_t>(0., size, 0., size),
		// default num
		0.,
		// const Lambda []
		[&](blocked_range2d<size_t> r, t running_total)
		{
			for (int i = r.rows().begin(); i < r.rows().end(); i++)
			for (int j = r.cols().begin(); j < r.cols().end(); j++)
			{
				if (matr1[i][j] < minValue)
					minValue = matr1[i][j];
				if (matr2[i][j] < minValue)
					minValue = matr2[i][j];
				if (matr1[i][j] > maxValue)
					maxValue = matr1[i][j];
				if (matr2[i][j] > maxValue)
					maxValue = matr2[i][j];
			}
	return running_total;
		},
		std::plus<double>());

	return omp_get_wtime() - t_start;
}

// Шаблонная функция вычисления значения суммы максимального и минимального элемента среди всех элементов двух исходных матриц tbb::parallel_reduce class
template <typename t>
double minmaxTbbClass(t**& matr1, t**& matr2, int size)
{
	double t_start = omp_get_wtime();

	MinMaxCalc<t> r(matr1, matr2);
	parallel_reduce(blocked_range2d<size_t>(0., size, 0., size), r);

	return omp_get_wtime() - t_start;
}

double AvgTrustedIntervalAVG(double*& times, int cnt)
{
	// вычисление среднеарифметического значения
	double avg = 0;
	for (int i = 0; i < cnt; i++)
	{
		// подсчет в переменную суммы
		avg += times[i];
	}
	// деление на количество
	avg /= cnt;
	// подсчет стандартного отклонения
	double sd = 0, newAVg = 0;
	int newCnt = 0;
	for (int i = 0; i < cnt; i++)
	{
		sd += (times[i] - avg) * (times[i] - avg);
	}
	sd /= (cnt - 1.0);
	sd = sqrt(sd);
	// вычисление нового среднего значения в доверительном интервале
	// с использованием среднеарифметического значения
	//
	for (int i = 0; i < cnt; i++)
	{
		if (avg - sd <= times[i] && times[i] <= avg + sd)
		{
			newAVg += times[i];
			newCnt++;
		}
	}
	if (newCnt == 0) newCnt = 1;
	return (newAVg / newCnt) * 1000;
}

template <typename t>
void calculationMatr()
{
	t** times = new t * [16];

	for (int i = 0; i < 16; i++)
		times[i] = new double[EXP_NUM];

	for (int size = 4250; size <= 8000; size += 1250)
	{
		std::cout << "Размер матриц " << size << std::endl; 
		t** matr1 = new t * [size],
			** matr2 = new t * [size];

		matr1[0] = new t[size * size];
		matr2[0] = new t[size * size];
		
		for (int i = 1; i < size; i++)
		{
			matr1[i] = &matr1[0][i * size];
			matr2[i] = &matr2[0][i * size];
		}

		fillMatr(matr1, size);
		fillMatr(matr2, size);

		for (int threads = 1; threads <= 4; threads++)
		{
			omp_set_num_threads(threads);

			// Установка количества потоков
			global_control
				global_limit(global_control::max_allowed_parallelism, threads);

			std::cout << "Число потоков = " << threads << " " << tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism) << std::endl;

			for (int i = 0; i < EXP_NUM; i++)
			{
				times[0][i] = addMatrixOmpParallel(matr1, matr2, size);
				times[1][i] = addMatrixOmpSections(matr1, matr2, size);
				times[2][i] = addMatrixTbbLambda(matr1, matr2, size);
				times[3][i] = addMatrixTbbClass(matr1, matr2, size);
				times[4][i] = multiplyMatrixOmpParallel(matr1, matr2, size);
				times[5][i] = multiplyMatrixOmpSections(matr1, matr2, size);
				times[6][i] = multiplyMatrixTbbLambda(matr1, matr2, size);
				times[7][i] = multiplyMatrixTbbClass(matr1, matr2, size);
				times[8][i] = summMatrixOmpParallel(matr1, matr2, size);
				times[9][i] = summMatrixOmpSections(matr1, matr2, size);
				times[10][i] = summMatrixTbbLambda(matr1, matr2, size);
				times[11][i] = summMatrixTbbClass(matr1, matr2, size);
				times[12][i] = minmaxOmpParallel(matr1, matr2, size);
				times[13][i] = minmaxOmpSections(matr1, matr2, size);
				times[14][i] = minmaxTbbLambda(matr1, matr2, size);
				times[15][i] = minmaxTbbClass(matr1, matr2, size);
			}
			std::cout << "Сложение матриц (OMP For, OMP Sections, TBB for_reduction lambda, TBB for_reduction class  - " << AvgTrustedIntervalAVG(times[0], EXP_NUM) << " " <<
				AvgTrustedIntervalAVG(times[1], EXP_NUM) << " " << AvgTrustedIntervalAVG(times[2], EXP_NUM) << " " << AvgTrustedIntervalAVG(times[3], EXP_NUM) << std::endl;
			std::cout << "Умножение матриц (OMP For, OMP Sections, TBB for_reduction lambda, TBB for_reduction class  - " << AvgTrustedIntervalAVG(times[4], EXP_NUM) << " " <<
				AvgTrustedIntervalAVG(times[5], EXP_NUM) << " " << AvgTrustedIntervalAVG(times[6], EXP_NUM) << " " << AvgTrustedIntervalAVG(times[7], EXP_NUM) << std::endl;
			std::cout << "Сумма всех элементов матриц (OMP For, OMP Sections, TBB for_reduction lambda, TBB for_reduction class  - " << AvgTrustedIntervalAVG(times[8], EXP_NUM) << " " <<
				AvgTrustedIntervalAVG(times[9], EXP_NUM) << " " << AvgTrustedIntervalAVG(times[10], EXP_NUM) << " " << AvgTrustedIntervalAVG(times[11], EXP_NUM) << std::endl;
			std::cout << "МинМакс матриц (OMP For, OMP Sections, TBB for_reduction lambda, TBB for_reduction class  - " << AvgTrustedIntervalAVG(times[12], EXP_NUM) << " " <<
				AvgTrustedIntervalAVG(times[13], EXP_NUM) << " " << AvgTrustedIntervalAVG(times[14], EXP_NUM) << " " << AvgTrustedIntervalAVG(times[15], EXP_NUM) << std::endl;
		}

		delete[] matr1[0], delete[] matr2[0], delete[] matr1, delete[] matr2;
	}
}

int main()
{
	setlocale(LC_ALL, "RUS");
	calculationMatr<double>();
	return 0;
}