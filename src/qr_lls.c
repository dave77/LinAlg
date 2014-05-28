/*
 * Linear Least Squares Solver
 *
 * Solves LLS using QR factorization.
 *
 *
 * Copyright David Wiltshire (c), 2014
 * All rights reserved
 *
 * Licence: see file LICENSE
 */

#include "qr_lls.h"

#include <assert.h>
#include <errno.h>
#include <float.h>
#include <math.h>

/* Calculate the norm of a vector */
static double 
norm(int n, const double *v)
{
	double res = 0.0;
	for (int i = 0; i < n; ++i)
		res += v[i] * v[i];
	return sqrt(res);
}

/* Solve a system of equations in upper triangular form */
static int
back_substitution(int m, int n, const double *restrict A, const double *restrict b,
		  double *x)
{
	int err = 0;
	for (int i = n - 1; i >= 0; --i) {
		double tmp = b[i];
		for (int j = i + 1; j < n; ++j) {
			tmp -= *(A + j * m + i) * x[j];
		}
		if (fabs(*(A + i * m + i)) < FLT_EPSILON) {
			err = -1;
			break;
		} else {
			x[i] = tmp / *(A + i * m + i);
		}
	}
	return err;
}

/*
 * Perform a Householder reflection of A and b, A an m x n matrix, b an
 * n vector.  The pivot is around Akk with the pivot being around the
 * column vector Ak + u0(e1) where e1 = (1 0 0 0 ... 0)
 */
static void 
householder_reflection(int m, int n, int k, double alpha, double u0, double *A, double *b)
{
	for (int j = k + 1; j < n; ++j) {
		double dot = u0 * *(A + j * m + k);
		for (int i = k + 1; i < m; ++i)
			dot += *(A + j * m + i) * *(A + k * m + i);
		dot /= fabs(alpha * u0);

		for (int i = k + 1; i < m; ++i)
			*(A + j * m + i) -= dot * *(A + k * m + i);
		*(A + j * m + k) -= dot * u0;
	}

	double dot = u0 * b[k];
	for (int i = k + 1; i < m; ++i) 
		dot += b[i] * *(A + k * m + i);
	dot /= fabs(alpha * u0);

	for (int i = k + 1; i < m; ++i)
		b[i] -= dot * *(A + k * m + i);
	b[k] -= dot * u0;
}

/* 
 * QR factorization: puts matrix A in upper triangular form (R) using QR
 * factorization and solves b = trans(Q) b simultaneously.
 */
static int
qr(int m, int n, double *restrict A, double *restrict b)
{
	assert (m >= n);
	
	for (int i = 0; i < n; ++i) {
		double col_norm = norm(m - i, (A + i * m + i));
		if (col_norm < FLT_EPSILON) // columns of A are (almost) linearly dependent
			return -1;
		double alpha = -copysignl(col_norm, *(A + i * m + i));	
		double u0 = *(A + i * m + i) + alpha;
		*(A + i * m + i) = -alpha;		
		householder_reflection(m, n, i, alpha, u0, A, b);
	}
	return 0;
}

/* QR solution of LLS problem */
int
qr_solve(int m, int n, double *restrict A, double *restrict b, double *restrict x)
{
	int err;
	if (m < n)
		return -EDOM;
	if ((err = qr(m, n, A, b)) < 0)
		return err;
	if ((err = back_substitution(m, n, A, b, x)) < 0)
		return err;
	return 0;
}
