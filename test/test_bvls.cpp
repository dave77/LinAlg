/*
 * Problems taken from:
 *	http://people.sc.fsu.edu/~jburkardt/f_src/qr_solve/qr_solve_prb_output.txt
 *	http://people.sc.fsu.edu/~jburkardt/f_src/bvls/bvls_prb_output.txt
 */
extern "C" {
#include "bvls.h"
}

#include "gtest/gtest.h"
#include <cfloat>

TEST(bvls, case1)
{
	static const int m = 2;
	static const int n = 2;

	double lb[n] = {1.0, 3.0};
	double ub[n] = {2.0, 4.0};
	double A[n][m] = {{0.965915, 0.747928},
			  {0.367391, 0.480637}};
	double b[m] = {0.997560, 0.566825};
	double x[n];
	double expected_x[n] = {1.00000, 3.00000};
	int rc;

	rc = bvls(m, n, (double *)A, b, lb, ub, x);
	
	ASSERT_EQ(0, rc);
	for (int i = 0; i < n; ++i) {
		EXPECT_NEAR(expected_x[i], x[i], 0.000001);
	}
}

TEST(bvls, case2)
{
	static const int m = 2;
	static const int n = 4;
	double A[n][m] = {{0.347081, 0.342244},
			  {0.217952, 0.133160},
			  {0.900524, 0.386766},
			  {0.445482, 0.661932}};
	double b[m] = {0.737543E-01, 0.535518E-02};
	double lb[n] = {0.0, 0.0, 0.0, 0.0};
	double ub[n] = {10.0, 10.0, 10.0, 10.0};
	double x[n];
	double expected_x[n] = {0.00000, 0.00000, 0.713028E-01, 0.00000};
	int rc;

	rc = bvls(m, n, (double *)A, b, lb, ub, x);
	
	ASSERT_EQ(0, rc);
	for (int i = 0; i < n; ++i) {
		EXPECT_NEAR(expected_x[i], x[i], 0.000001);
	}
}

TEST(bvls, case3)
{
	static const int m = 4;
	static const int n = 2;
	double A[n][m] = {{0.855692, 0.401287, 0.206874, 0.968539},
			  {0.598400, 0.672981, 0.456882, 0.330015}};
	double b[m] = {0.161083E-01, 0.650855, 0.646409, 0.322987};
	double lb[n] = {0.0, -100.0};
	double ub[n] = {100.0, 100.0};
	double x[n];
	double expected_x[n] = {0.00000, 0.752745};
	int rc;
	rc = bvls(m, n, (double *)A, b, lb, ub, x);
	
	ASSERT_EQ(0, rc);
	for (int i = 0; i < n; ++i) {
		EXPECT_NEAR(expected_x[i], x[i], 0.000001);
	}
}

TEST(bvls, case5)
{
	static const int n = 5;
	static const int m = 10;
	double A[n][m] = {{0.114244, 0.318463, 0.596820, 0.481529E-01, 0.114206,
			   0.215965, 0.100573, 0.733418E-01, 0.246862, 0.443384},
			  {0.208368, 0.566998, 0.243124E-01, 0.420291, 0.397853,
			   0.976585, 0.692605, 0.494331E-02, 0.129921, 0.467772E-01},
			  {0.839778, 0.678489, 0.581951, 0.733526, 0.116043, 
			   0.840300, 0.834996, 0.746536, 0.843201, 0.528839}, 
			  {0.665485, 0.730737, 0.410604, 0.355722, 0.735377,
			   0.471318, 0.462625, 0.759692, 0.702459, 0.257966},
			  {0.937705, 0.456104, 0.808489, 0.908848, 0.694877,
			   0.219489, 0.854955, 0.744397, 0.301113, 0.671969}    
			};
	double b[m] = {0.261575, 0.765595E-01, 0.101250, 0.549266, 0.375585,   
		       0.151495E-01, 0.792915, 0.620878, 0.773604, 0.953581};
	double lb[n] = {0.00000, -1.00000, 0.00000, 0.300000, 0.480000E-01};
	double ub[n] = {1.00000,  0.00000, 1.00000, 0.400000, 0.490000E-01};
	double x[n];
	double expected_x[n] = {0.0, -0.352785, 0.521598, 0.3, 0.049}; 
	int rc;
	rc = bvls(m, n, (double *)A, b, lb, ub, x);
	
	ASSERT_EQ(0, rc);
	for (int i = 0; i < n; ++i) {
		EXPECT_NEAR(expected_x[i], x[i], 0.000001);
	}
}

TEST(bvls, case6)
{
	static const int m = 6;
	static const int n = 4;
	double lb[n] = {-100.000, -FLT_MAX, -FLT_MAX, -FLT_MAX};
	double ub[n] = {100.000, FLT_MAX, FLT_MAX, FLT_MAX};
	double A[n][m] = {{0.930943, 0.546529, 0.465513, 0.176914, 0.377994, 0.175221},
			  {0.356858, 0.665676, 0.618549, 0.468947, 0.316252, 0.268880},
			  {0.199921, 0.415428, 0.979303, 0.543212, 0.129518, 0.804742},
			  {0.845821, 0.709033, 0.528973, 0.663841, 0.900739, 0.941164}};
	double b[m] = {0.618714, 0.967574, 0.990239, 0.338006, 0.920767, 0.339073}; 
	double x[n];
	double expected_x[n] = {0.224131, 1.24988, -0.189694, 0.173490};
	int rc;
	rc = bvls(m, n, (double *)A, b, lb, ub, x);
	
	ASSERT_EQ(0, rc);
	for (int i = 0; i < n; ++i) {
		EXPECT_NEAR(expected_x[i], x[i], 0.00001);
	}
}

TEST(bvls, qr_test4)
{
	static const int N = 3;
	static const int M = 5;
	double A[N][M] =
	     {{1, 1, 1, 1, 1},
	     {1, 2, 3, 4, 5},
	     {1, 4, 9, 16, 25}};
	double b[M] = {1, 2.3, 4.6, 3.1, 1.2};
	double x[N];
	double lb[N] = {-10, -10, -10};
	double ub[N] = {10, 10, 10};
	double expected_x[N] = {-3.0200000, 4.4914286, -0.72857143};
	int rc;
	rc = bvls(M, N, (double *)A, b, lb, ub, x);

	ASSERT_EQ(0, rc);
	for (int i = 0; i < N; ++i) {
		EXPECT_NEAR(expected_x[i], x[i], 0.00001);
	}
 
}
#if 0

 M =    5,   N =   10,   UNBND =      0.10000E+07
 
  Bounds:
 
 
     0.00000     -0.399400      -1.00000     -0.300000       21.0000    
     0.00000     -0.399400       1.00000     -0.200000       22.0000    
 
    -4.00000       45.0000       100.000     -0.340282E+39  -1.00000    
    -3.00000       46.0000       101.000      0.340282E+39   1.00000    
 
  Matrix A:
 
 
    0.658229      0.256798      0.901923      0.147835      0.614369    
    0.150717      0.550865      0.657925      0.674529      0.820617    
    0.612315      0.659047      0.728858      0.769614      0.947095    
    0.978660      0.554005      0.402455      0.339323      0.731129    
    0.999142      0.977760      0.928628      0.115819      0.497604    
 
    0.374802      0.746310      0.946848      0.480381      0.885878    
    0.421506      0.953759      0.706176      0.597690      0.303810    
    0.552903      0.932747E-01  0.813810      0.137532      0.669657    
    0.997919      0.734024      0.558594      0.587395      0.664940    
    0.990395      0.751762      0.617055E-01  0.519968      0.503677    
 
  RHS B:
 
    0.100383      0.755453      0.605693      0.719048      0.897335    
 
  After BVLS:  No. of components not at constraints =       1
 
  Solution vector, X:
 
     0.00000     -0.399400      -1.00000     -0.300000       21.0000    
    -3.00000       46.0000       100.000      -196.029      -1.00000   



int problem2(void)
{
	double A[N][M] = 
	     {{1, 1, 1, 1, 1},
	     {1, 2, 3, 4, 5},
	     {1, 4, 9, 16, 25}};
 
	double b[M] = {1, 2.3, 4.6, 3.1, 1.2};
	double x[N];
	double lb[N] = {-10, -10, -10};
	double ub[N] = {10, 10, 10};
	int rc;
	
}

	double r[M];
	double rnorm = 0.0;
	for (int i = 0; i < M; ++i) {
		r[i] = b[i];
		for (int j = 0; j < N; ++j)
			r[i] -= A[j][i] * x[j];
		rnorm += r[i] * r[i];
	}
#endif
