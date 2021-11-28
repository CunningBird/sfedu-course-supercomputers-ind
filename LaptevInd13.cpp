#pragma warning(disable : 4996)

#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include "omp.h"

const int N = 360;

void matrix_write_file(const char* fname);
int** read_file(const char* fname);
int* matrix_convert_to_vec(const size_t sz_b, int** A);
int* sort_matrix_by_blocs(const size_t sz_b, int* A);
void matrix_free(int **A);

int* mulpiplication(int* A, int* B, const size_t sz_b);
int* mulpiplication_parallel(int* A, int* B, const size_t sz_b);
int* mulpiplication_parallel_reorder(int* A, int* B, const size_t sz_b);
int* get_block(int* A, int sz_b);

int main(int argc, char**argv)
{
	matrix_write_file("upper.txt");
	int** u_mtrx = read_file("upper.txt");
	matrix_write_file("lower.txt");
	int** l_mtrx = read_file("lower.txt");

	for (size_t sz_b(1); sz_b <= N; ++sz_b)
	{
		if (N%sz_b == 0)
		{
			FILE* dfile = fopen("dividers.txt", "a");
			fprintf(dfile, "%d\n", sz_b);
			printf("\nblock size: %d\n", sz_b);

			int* A = matrix_convert_to_vec(sz_b, u_mtrx);
			int* B = matrix_convert_to_vec(sz_b, l_mtrx);
			int* C1 = mulpiplication(A, B, sz_b);
			delete[] C1;
			int* C2 = mulpiplication_parallel(A, B, sz_b);
			delete[] C2;
			int* C3 = mulpiplication_parallel_reorder(A, B, sz_b);
			delete[] C3;

			delete[] A;
			delete[] B;

			fclose(dfile);
		}
	}
	matrix_free(u_mtrx);
	matrix_free(l_mtrx);
	printf("\n");
	return 0;
}

void matrix_write_file(const char* fname) {
	int** A = new int*[N];
	for (size_t i(0); i < N; i++) {
		A[i] = new int[N];
	}

	for (size_t i(0); i < N; ++i) {
		for (size_t j(0); j < N; ++j) {
		    A[i][j] = rand() % 10;
		}
	}

	FILE* file = fopen(fname, "w");

	for (size_t i(0); i < N; ++i) {
		for (size_t j(0); j < N; ++j) {
			fprintf(file, "%4d", A[i][j]);
		}
		fprintf(file, "\n");
	}

	fclose(file);
	matrix_free(A);
}

int* sort_matrix_by_blocs(const size_t sz_b, int* A) {

	int* A1 = new int[N*N];

	#pragma omp parallel num_threads(4)
	{
		size_t S = N / sz_b;

		#pragma omp for  schedule(static)
		for (int i(0); i < S; ++i) {
			for (int j(0); j < S; ++j) {
				for (int k(j); k < S; ++k) {
					int *a = A + (i * (S + 1) - (i + 1)*i / 2 + (k - i)) * sz_b*sz_b;
					int *a1 = A1 + i * N*sz_b + j * sz_b*sz_b;

					for (int ii(0); ii < sz_b; ++ii) { 
						for (int jj(0); jj < sz_b; ++jj) {
							for (int kk(0); kk < sz_b; ++kk) {
								a1[ii + jj] += a[ii*sz_b + kk];
							}
						}
					}
				}
			}
		}
	}

	return A1;
}

void matrix_free(int **A) {
	for (size_t i(0); i < N; i++)
		delete[] A[i];

	delete[] A;
}

int* matrix_convert_to_vec(const size_t sz_b, int** A) {
	size_t S(N / sz_b);
	int* vec = new int[(S + 1)*S / 2 * sz_b*sz_b];
	size_t i = 0, j = 0, t0 = 0, t1 = 0, k = 0;

	for (size_t t0(0); t0 < S; ++t0) {
		for (size_t t1(t0); t1 < S; ++t1) {
			for (size_t i(0); i < sz_b; ++i) {
				for (size_t j(0); j < sz_b; ++j) {
				    k++
					vec[k] = A[i + t1 * sz_b][j + t0 * sz_b];
					vec[k] = A[i + t0 * sz_b][j + t1 * sz_b];
				}
			}
		}
	}

	return vec;
}

int** read_file(const char* fname) {
	int** A = new int*[N];
	for (size_t i(0); i < N; i++) {
		A[i] = new int[N];
	}

	FILE *file;
	file = fopen(fname, "r");

	for (size_t i(0); i < N; ++i) {
		for (size_t j(0); j < N; ++j) {
			if (!feof(file)) fscanf(file, "%d", &A[i][j]);
		}
	}

	fclose(file);
	
	return A;
}

int* mulpiplication(int* A, int* B, size_t sz_b) {
	int* C = new int[N*N];
	for (size_t i(0); i < N*N; ++i) {
		C[i] = 0;
	}

	double t1, t2;
	t1 = omp_get_wtime();
	size_t S = N / sz_b;

	for (size_t i(0); i < S; ++i) {
		for (size_t j(0); j < S; ++j) {
			for (size_t k(j); k < S; ++k) {
				int *a = A + (i * (S + 1) - (i + 1)*i / 2 + (k - i)) * sz_b*sz_b,
					*b = B + (j * (S + 1) - (j + 1)*j / 2 + (k - j)) * sz_b*sz_b,
					*c = C + i * N*sz_b + j * sz_b*sz_b;

				for (size_t ii(0); ii < sz_b; ++ii) {
					for (size_t jj(0); jj < sz_b; ++jj) {
						for (size_t kk(0); kk < sz_b; ++kk) {
							c[ii*sz_b + jj] += a[ii*sz_b + kk] * b[kk* sz_b + jj];
						}
					}
				}
			}
		}
	}

	t2 = omp_get_wtime();

	FILE* file = fopen("time.txt", "a");
	fprintf(file, "%f\n", (t2 - t1));
	fclose(file);

	return C;
}

int* mulpiplication_parallel(int* A, int* B, size_t sz_b) {
	int* C = new int[N*N];
	for (size_t i(0); i < N*N; ++i) {
		C[i] = 0;
	}

	double t1, t2;
	t1 = omp_get_wtime();

	#pragma omp parallel num_threads(12) 
	{
		size_t S = N / sz_b;
		#pragma omp for  schedule(static)
		for (int i(0); i < S; ++i) { // Обход строки блоков
			for (int j(0); j < S; ++j) { // Обход столбца блоков
				for (int k(j); k < S; ++k) { // Перемножение двух блоков (Подготовка, начало умножения)
					int *a = A + (i * (S + 1) - (i + 1)*i / 2 + (k - i)) * sz_b*sz_b,
						*b = B + (j * (S + 1) - (j + 1)*j / 2 + (k - j)) * sz_b*sz_b,
						*c = C + i * N*sz_b + j * sz_b*sz_b;

					for (int ii(0); ii < sz_b; ++ii) {
						for (int jj(0); jj < sz_b; ++jj) {
							for (int kk(0); kk < sz_b; ++kk) {
								c[ii*sz_b + jj] += a[ii*sz_b + kk] * b[kk* sz_b + jj];
							}
						}
					}
				}
			}
		}
	}
	
	t2 = omp_get_wtime();;

	FILE* file = fopen("time_parallel.txt", "a");
	fprintf(file, "%f\n", (t2 - t1));
	fclose(file);

	return C;
}

int* mulpiplication_parallel_reorder(int* A, int* B, size_t sz_b) {
	int* A1 = sort_matrix_by_blocs(sz_b, A);
	int* B1 = sort_matrix_by_blocs(sz_b, B);

	int* C = new int[N*N];
	for (size_t i(0); i < N*N; ++i) {
		C[i] = 0;
	}

	double t1, t2;
	t1 = omp_get_wtime();

	#pragma omp parallel num_threads(12)
	{
		size_t S = N / sz_b;
		#pragma omp for  schedule(static)
		for (int i(0); i < S; ++i) {
			for (int j(0); j < S; ++j) {
				for (int k(j); k < S; ++k) {
					int *a = A1 + (i * (S + 1) - (i + 1)*i / 2 + (k - i)) * sz_b*sz_b;
					int *b = B1 + (j * (S + 1) - (j + 1)*j / 2 + (k - j)) * sz_b*sz_b;
					int *c = C + i * N*sz_b + j * sz_b*sz_b;

					for (int ii(0); ii < sz_b; ++ii) {
						for (int jj(0); jj < sz_b; ++jj) {
							for (int kk(0); kk < sz_b; ++kk) {
								c[ii + jj] += a[ii + kk] * b[kk + jj];
							}
						}
					}
				}
			}
		}
	}

	t2 = omp_get_wtime();

	FILE* file = fopen("time_parallel_divide.txt", "a");
	fprintf(file, "%f\n", (t2 - t1));
	fclose(file);

	return C;
}