/* Bounded Variable Least Squares
 *
 * Solves the Bounded Variable Least Squares problem (or box-constrained
 * least squares) of:
 *	minimize   ||Ax - b||
 *	   x		     2
 *
 *	given lb <= x <= ub
 *
 *	where:
 *		A is an m x n matrix
 *		b, x, lb, ub are n vectors
 *
 * Based on the article Stark and Parker "Bounded-Variable Least Squares: an Alogirthm
 * and Applications" retrieved from: http://www.stat.berkeley.edu/~stark/Preprints/bvls.pdf
 *
 * Copyright David Wiltshire (c), 2014
 * All rights reserved
 *
 * Licence: see file LICENSE
 */

// Local Includes
#include "bvls.h"
#include "qr_lls.h"

// Standard Library Includes
#include <assert.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/*
 * Computes w(*) = trans(A)(Ax -b), the negative gradient of the
 * residual.
 */
static void
negative_gradient(int m, int n, const double *restrict A, const double *restrict b,
		  const double *restrict x, double *restrict w)
{
	// Initialize w
	for (int j = 0; j < n; ++j)
		w[j] = 0.0;

	for (int i = 0; i < m; ++i) {
		double ri = b[i];
		for (int j = 0; j < n; ++j)
			ri -= *(A + j * m + i) * x[j];
		for (int j = 0; j < n; ++j)
			w[j] += *(A + j * m + i) * ri;
	}
}

/* Find the index which most wants to be free.  Or return -1 */
static int
find_index_to_free(int n, const double *w, const int8_t *istate)
{
	int index = -1;
	double max_grad = 0.0;

	for (int i = 0; i< n; ++i) {
		double gradient = -w[i] * istate[i];
		if (gradient > max_grad) {
			max_grad = gradient;
			index = i;
		}
	}
	return index;
}

/* Move index to the free set */
static void
free_index(int index, int8_t *istate, int *indices, int *num_free)
{
	assert(index >= 0);
	istate[index] = 0;
	indices[*num_free] = index;
	++(*num_free);
}

/*
 * Build matrix A' and b' where A' is those columns of A that are free
 * and b' is the vector less the contribution of the bound variables
 */
static void
build_free_matrices(int m, int n, int num_free, const double *restrict A,
		    const double *restrict b, const int *restrict indices,
		    const int8_t *istate, const double *restrict x,
		    double *restrict act_A, double *restrict act_b)
{
	/* Set A' to free columns of A */
	for (int i = 0; i < num_free; ++i) {
		memcpy(act_A, (A + m * indices[i]), m * sizeof(*act_A));
		act_A += m;
	}

	/* Set b' = b */
	memcpy(act_b, b, m * sizeof(*act_b));
	/* Adjust b'j = bj - sum{Aij * x[j]} for i not in Free set */
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (istate[j] != 0) {
				act_b[i] -= *(A + j * m + i) * x[j];
			}
		}
	}
}

/* Check that suggested solution is with in bounds */
static bool
check_bounds(int num_free, const int *indices, const double *restrict z,
	     const double *restrict lb, const double *restrict ub)
{
	for (int i = 0; i < num_free; ++i) {
		int index = indices[i];
		if (z[i] < lb[index] || z[i] > ub[index])
			return false;
	}
	return true;
}

/*
 * Set the solution vector to suggested solution ... call only after
 * checking bounds!
 */
static void
set_x_to_z(int num_free, const int *indices, const double *restrict z, double *restrict x)
{
	for (int i = 0; i < num_free; ++i)
		x[indices[i]] = z[i];
}

static void
bind_index(int index, bool up, double fixed, double *x, int *num_free, int *indices,
	   int8_t *istate)
{
	x[indices[index]] = fixed;
	istate[indices[index]] = up ? 1 : -1;
	memmove(indices + index, indices + index + 1, *num_free - index - 1);
	--(*num_free);
}

static int
find_index_to_bind(int *num_free, int *indices, const double *restrict z,
		   double *restrict x, const double *restrict lb,
		   const double *restrict ub, int8_t *istate)
{
	int index = -1;
	bool bind_up = false;
	double alpha = DBL_MAX;

	for (int i = 0; i < *num_free; ++i) {
		int ii = indices[i];
		double interpolate;
		if (z[i] <= lb[ii]) {
			interpolate = (lb[ii] - x[ii]) / (z[i] - x[ii]);
			if (interpolate < alpha) {
				alpha = interpolate;
				index = i;
				bind_up = false;
			}
		} else if (z[i] >= ub[ii]) {
			interpolate = (ub[ii] - x[ii]) / (z[i] - x[ii]);
			if (interpolate < alpha) {
				alpha = interpolate;
				index = i;
				bind_up = true;
			}
		}
	}

	assert(index >= 0);

	for (int i = 0; i < *num_free; ++i) {
		int ii = indices[i];
		x[ii] += alpha * (z[i] - x[ii]);
	}

	double limit = bind_up? ub[indices[index]] : lb[indices[index]];
	bind_index(index, bind_up, limit, x, num_free, indices, istate);
	return index;
}

/* Move variables that are out of bounds to their respective bound */
static void
adjust_sets(int *num_free, const double *lb, const double *ub, double *x,
	    int *indices, int8_t *istate)
{
	for (int ii = 0; ii < *num_free; ++ii) {
		int i = indices[ii];
		if (x[i] <= lb[i]) {
			bind_index(i, false, lb[i], x, num_free, indices, istate);
		} else if (x[i] >= ub[i]) {
			bind_index(i, true, ub[i], x, num_free, indices, istate);
		}
	}
}

/* Allocate working arrays */
static int
allocate(int m, int n, double **w, double **act_A, double **z,
		int8_t **istate, int **indices)
{
	*w = malloc(n * sizeof(**w));
	*act_A = malloc(m * n * sizeof(**act_A));
	*z = malloc(m * sizeof(**z));
	*istate = malloc(n * sizeof(**istate));
	*indices = malloc(n * sizeof(**indices));

	if (*w       == NULL ||
	    *act_A   == NULL ||
	    *z       == NULL ||
	    *istate  == NULL ||
	    *indices == NULL)
		return -ENOMEM;
	else
		return 0;
}

/* Free memory */
static void
clean_up(double *w, double *act_A, double *z, int8_t *istate, int *indices)
{
	free(w);
	free(act_A);
	free(z);
	free(istate);
	free(indices);
}

/*
 * Initializes the problems: sets x[i] = lb[i] and sets istate and
 * indices correctly
 */
static int
init(int n, const double *restrict lb, const double *restrict ub, double *restrict x,
		int8_t *istate, int *indices)
{
	for (int i = 0; i < n; ++i) {
		if (lb[i] > ub[i])
			return -1;
		x[i] = lb[i];
		indices[i] = -1;
		istate[i] = -1;
	}
	return 0;
}

/* The BVLS main function */
int
bvls(int m, int n, const double *restrict A, const double *restrict b,
	const double *restrict lb, const double *restrict ub, double *restrict x)
{
	double *w;
	double *act_A;
	double *z;
	int8_t *istate;
	int *indices;
	int num_free = 0;
	int rc;
	int prev = -1;

	rc = allocate(m, n, &w, &act_A, &z, &istate, &indices);
	if (rc < 0)
		goto out;
	rc = init(n, lb, ub, x, istate, indices);
	if (rc < 0)
		goto out;

	negative_gradient(m, n, A, b, x, w);
	for (int i = 0; i < 3 * n; ++i) {
		int index_to_free = find_index_to_free(n, w, istate);
		/*
		 * If no index on a bound wants to move in to the
		 * feasible region then we are done
		 */
		if (index_to_free < 0)
			break;

		if (index_to_free == prev) {
			w[prev] = 0.0;
			continue;
		}

		/* Move index to free set */
		free_index(index_to_free, istate, indices, &num_free);
		/* Solve Problem for free set */
		build_free_matrices(m, n, num_free, A, b, indices, istate, x, act_A, z);
		rc = qr_solve(m, num_free, act_A, z);
		if (rc < 0) {
			prev = index_to_free;
			w[prev] = 0.0;
			if (x[prev] == lb[prev]) {
				istate[prev] = -1;
			} else {
				istate[prev] = 1;
			}
			--num_free;
			continue;
		}

		if (check_bounds(num_free, indices, z, lb, ub)) {
			set_x_to_z(num_free, indices, z, x);
			prev = -1;
		} else {
			prev = find_index_to_bind(&num_free, indices, z, x, lb, ub, istate);
			adjust_sets(&num_free, lb, ub, x, indices, istate);
			while (num_free > 0) {
				build_free_matrices(m, n, num_free, A, b, indices, istate, x, act_A, z);
				rc = qr_solve(m, num_free, act_A, z);
				// so something is very wrong, we must
				// have solved these columns before so
				// rc must be 0.
				assert(rc == 0);
				if (check_bounds(num_free, indices, z, lb, ub)) {
					set_x_to_z(num_free, indices, z, x);
					break;
				} else {
					prev = find_index_to_bind(&num_free, indices, z, x, lb, ub, istate);
					adjust_sets(&num_free, lb, ub, x, indices, istate);
				}
			}
		}
		negative_gradient(m, n, A, b, x, w);
	}
out:
	clean_up(w, act_A, z, istate, indices);
	return rc;
}
