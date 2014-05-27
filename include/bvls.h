#ifndef BVLS_H_
#define BVLS_H_

/** Bounded Variable Least Squares Solver.
 * 
 * Solves the problem:
 *
 * 	     minimize ||Ax - b||
 *		x		2
 *	     with lb <= x <= ub
 *
 * Where
 *	A is an n x m matrix
 *	b is a m vector
 *	lb, ub are n vectors
 *	x is an n vector
 *
 * @param m [in] number of columns
 * @param n [in] number of rows
 * @param A [in] coefficient Matrix 
 * @param b [in] target vector
 * @param lb [in] lower bounds
 * @param ub [in] upper bounds
 * @param x [out] result vector  
 *
 * @return 0 if successfuly, < 0 on error
 */
int bvls(int m, int n,
	 const double *restrict A, 
	 const double *restrict b,
	 const double *restrict lb,
	 const double *restrict ub,
	 double *restrict x);

#endif // BVLS_H_
