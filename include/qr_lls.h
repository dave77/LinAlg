/* Linear Least Squares Solver, using QR factorization
 *
 * Copyright David Wiltshire (c), 2014
 * All rights reserved
 *
 * Licence: see file LICENSE
 */

#ifndef QR_LLS_H_
#define QR_LLS_H_

/** Solves a linear least squares problem using QR factorization.
 *
 * Solves the problem:
 *	minimize	||Ax - b||
 *	   x			  2
 *	where
 *		A is an m x n matrix
 *		b,x are n vectors
 *
 * This function overwrites A and b in the process of solving for x (the
 * solution vector).  The method use a QR factorization of the matrix A
 * to solve the LLS problem.  Returns an error if the problem is under
 * constrained (i.e. m < n).
 *
 * @param m	number of rows of A, b and x
 * @param n	number of columns of A
 * @param A	m x n matrix stored in column order (ie Aij isA [j][i])
 * @param b	n vector of data points
 * @param x	n vector of the solution.
 *
 * @return	0 if successful, < 0 on error
 */
int qr_solve(int m, int n, double *restrict A, double *restrict b, double *restrict x);

#endif
