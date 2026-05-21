#ifndef PUSHING_H
#define PUSHING_H

//#define VERBOSE
//#define DEBUG


// HEADER
// ======
#include <stdint.h>

/*
**Description:**
Constructs a pushing triangulation/fan of the input vectors. Optionally,
- random or
- random & fine.
These configurations are set by user-specified arguments in PushingOpts.

**Arguments:**
- `vecs`:          The input vectors.
- `dim`:           The dimension of the vector configuration.
- `num_vecs`:      The number of vectors input.
// configuration
- `opts`:          Configuration for algorithm. Whether to output a
                   triangulation using the input vector order, a random
                   triangulation, or a random & fine triangulation.
// simplices objects
- `max_num_simps`: Max allowed number of simplices - to prevent writing out
                   of simps container.
- `simps`:         OUTPUT: A container for the simplices.
- `num_simps`:     OUTPUT: The number of simplices written.

**Returns:**
A status code according to following list:
    0:  success
    -1: misconfigured options. If fine=1, need random=1.
    -2: memory allocation problem
    -3: 0 vector input
    -4: couldn't find initial simplex
    -5: deadlock state - couldn't add a new simplex
    -6: constructed too many simplices
    -100: error in splitting a cone - see code
*/
typedef struct {
    int random;
    int fine;
    uint64_t seed;
} PushingOpts;

int pushing(
    int *vecs,
    int dim,
    int num_vecs,
    PushingOpts *opts,
    int max_num_simps,
    uint32_t *simps,
    int *num_simps
);


// IMPLEMENTATION
// ==============
#ifdef PUSHING_IMPLEMENTATION

#ifndef MAX_DIM
#define MAX_DIM 8
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// optional hardcoded determinants
#if defined(__has_include) && __has_include("det.h")
#include "det.h"
#endif


// EXTERNAL METHODS (copied and then modified a bit for syntax/similar)
// ----------------
/*  Written in 2019 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. */

/* This is xoshiro256++ 1.0, one of our all-purpose, rock-solid generators.
   It has excellent (sub-ns) speed, a state (256 bits) that is large
   enough for any parallel application, and it passes all tests we are
   aware of.

   For generating just floating-point numbers, xoshiro256+ is even faster.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill rng_state. */

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t next(uint64_t rng_state[4]) {
    const uint64_t result = rotl(rng_state[0] + rng_state[3], 23) + rng_state[0];

    const uint64_t t = rng_state[1] << 17;

    rng_state[2] ^= rng_state[0];
    rng_state[3] ^= rng_state[1];
    rng_state[1] ^= rng_state[2];
    rng_state[0] ^= rng_state[3];

    rng_state[2] ^= t;

    rng_state[3] = rotl(rng_state[3], 45);

    return result;
}
/* End of xoshiro256++ by Blackman and Vigna */

// for seeding xoshiro256++
static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

// Fisher-Yates (shuffle list)
static void fisher_yates(int * lis, int len, uint64_t* rng_state) {
    uint64_t j;

    for (int i=len-1; i>=0; --i) {
        j = next(rng_state) % (i+1); // get a random index to swap
        int tmp = lis[i]; lis[i] = lis[j]; lis[j] = tmp;
    }
}

// PUSHING CODE
// ============
// HELPER DATA STRUCTURES
// ----------------------
typedef struct {
    int labels[MAX_DIM];              // vertices of the simplex
    int normals[MAX_DIM*MAX_DIM];     // inwards-facing normals
    int external_facet_inds[MAX_DIM]; // indices of external facets... unsorted
    int num_external_facets;
} Simplex;

// MISC CUSTOM HELPERS
// -------------------
static void insertion_sort(int *arr, int len) {
    for (int i = 1; i < len; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j+1] = arr[j];
            j--;
        }
        arr[j+1] = key;
    }
}

static int gcd(int a, int b) {
    a = abs(a); b = abs(b);
    while (b) { int t = b; b = a % b; a = t; }
    return a;
}

static void reduce_by_gcd(int *v, int dim) {
    // reduces a vector by its gcd
    int g = 0;
    for (int i = 0; i < dim; i++) g = gcd(g, v[i]);
    if (g <= 1) return;
    for (int i = 0; i < dim; i++) v[i] /= g;
}

static int det(int *M, int dim) {
    // computes the determinant of M, a dim-by-dim matrix

    // base case
    switch (dim) { 
        #ifdef DET1
        case 1: return det1(M);
        #endif

        #ifdef DET2
        case 2: return det2(M);
        #endif

        #ifdef DET3
        case 3: return det3(M);
        #endif

        #ifdef DET4
        case 4: return det4(M);
        #endif

        #ifdef DET5
        case 5: return det5(M);
        #endif

        #ifdef DET6
        case 6: return det6(M);
        #endif

        default: ;
    }
    // in case there is no hardcoded det.h file, here is a base case
    if (dim == 1)
        return M[0];

    // recurse
    int out = 0;
    int M_trim[(dim-1)*(dim-1)];
    int sign = -1;
    for (int i=0; i<dim; ++i) {
        // update the sign
        sign *= -1;

        // trim ith row and 0th column
        for (int itrim=0; itrim<dim-1; ++itrim) {
            int jumped = (itrim >= i); // whether we skipped row-i
            for (int jtrim=0; jtrim<dim-1; ++jtrim) {
                M_trim[(dim-1)* itrim+jtrim] = M[dim* (itrim+jumped)+(jtrim+1)];
            }
        }

        // Laplace expansion
        out += sign*M[dim* i+0]*det(M_trim, dim-1);
    }

    return out;
}

static int dot(int *a, int *b, int dim) {
    int out = 0;
    for (int i=0; i<dim; ++i)
        out += a[i]*b[i];
    return out;
}

// H-REP OF N-1 SIMPLEX
// --------------------
static void simp_facet_normal(int *R, int dim, int *x) {
    // for R the (row-wise) rays of a facet of a simplex, compute the normal x
    // i.e., the vector x such that Rx=0.
    // does so by setting x=(-1)^k det(R[:,:!=i])

    int R_trim[(dim-1)*(dim-1)];
    int sign = -1;
    for (int xi=0; xi<dim; ++xi) {
        // update the sign
        sign *= -1;

        // get the trimmed arr - skipping column-xi
        for (int jtrim=0; jtrim<dim-1; ++jtrim) {
            int jumped = (jtrim >= xi); // whether we skipped column-xi
            for (int itrim=0; itrim<dim-1; ++itrim) {
                R_trim[(dim-1)* itrim+jtrim] = R[dim* itrim+(jtrim+jumped)];
            }
        }

        // set x[i]
        x[xi] = sign*det(R_trim,dim-1);
    }

    // reduce normal by GCD...
    reduce_by_gcd(x, dim);
}

static void hrep(int *R, int dim, int *H) {
    // get the inwards-facing H-representation of the simplex
    // (iterate over all facets, get the normal, orient so it faces inwards)
    // facets come in order R\i for i=0,1,2,...

    int R_facet[(dim-1)*dim];
    for (int i=0; i<dim; ++i) {
        // get the rays corresponding to this facet
        for (int ifacet=0; ifacet<dim-1; ++ifacet) {
            int jumped = (ifacet >= i);
            for (int jfacet=0; jfacet<dim; ++jfacet) {
                R_facet[dim* ifacet+jfacet] = R[dim* (ifacet+jumped)+jfacet];
            }
        }

        // get the normal
        simp_facet_normal(R_facet, dim, &H[i*dim]);

        // ensure it is inwards-facing
        int dotted = dot(&H[dim*i],&R[dim*i],dim);

        if (dotted<0) {
            for (int k=0; k<dim; ++k)
                H[i*dim+k] *= -1;
        }

    }
}

static int simp_contains(
    Simplex *simp,
    int *vecs,
    int dim,
    int *labels,
    int num_labels,
    int stop_at_first_containment,
    int *contained_labels) {
    /*
    Computes which of the dim-dimensional vectors (as specified by the labels)
    is contained in the input simplex. Ignores the labels explicitly included
    in the simplex.

    Optionally, allow stopping at the first containment.

    Returns the number of containments found so far. If this is 0, then no other
    vectors are contained in simp.
    */
    int num_contained = 0;

    for (int ilabel=0; ilabel<num_labels; ++ilabel){
        int label = labels[ilabel];

        // skip if label is explicitly in simp
        int skip = 0;
        for (int isimp=0; isimp<dim; ++isimp) {
            if (label == simp->labels[isimp]) {skip = 1; break;}
        }
        if (skip == 1) continue;

        // check if label is in conical hull
        int bad = 1;
        int dotted;
        for (int ifacet=0; ifacet<dim; ++ifacet) {
            dotted = dot(
                &(simp->normals[MAX_DIM*ifacet]),
                &vecs[dim*label],
                dim);

            bad = bad && (dotted>=0);
        }

        // label *is* in conical hull :(
        if (bad) {
            contained_labels[num_contained++] = label;
            if (stop_at_first_containment)
                return num_contained;
        }
    }

    return num_contained;
}

// PUSHING BEGINS
// ==============
int pushing(
    int *vecs,
    int dim,
    int num_vecs,
    PushingOpts *opts,
    int max_num_simps,
    uint32_t *simps,
    int *num_simps
)
{
    // input checking
    if (opts->fine && (opts->random == 0)) {
        return -1;
    }

    // set up some variables
    // ---------------------
    // misc
    int return_code = 0;
    uint64_t rng_state[4]; // RNG state

    // vc variables
    int num_labels = num_vecs;
    int labels[num_vecs]; // labels, defined as 0,1,...,num_vecs-1
    for (int i = 0; i < num_vecs; i++) { labels[i] = i; }

    // simplex variables
    int _inds[dim]; // indices into the (shuffled) labels... defines simplex
    int simp_labels[dim];
    int seed_simp_V[dim*dim];
    int seed_simp_H[dim*dim];

    int contained_labels[num_labels];
    int num_contained;

    // fan variables
    Simplex *_simps     = NULL;
    int visible_numfacets;
    int *visible_isimp  = NULL;
    int *visible_ifacet = NULL;

    // initialize the fan variables
    *num_simps      = 0;
    _simps = malloc(max_num_simps * sizeof(Simplex)); // internal use
    if (_simps == NULL) { return_code = -2; goto end; }

    // input checks
    // ------------
    // (reject inputs if vector of all 0s)
    #ifdef VERBOSE
    printf("Input vectors:");
    #endif
    for (int ivec=0; ivec<num_vecs; ++ivec) {
        #ifdef VERBOSE
        printf("[");
        #endif

        int _vi;
        int bad = 1;
        for (int j=0; j<dim; ++j) {
            _vi = vecs[dim* ivec+j];
            bad = bad && (_vi==0);

            #ifdef VERBOSE
            printf("%d,",_vi);
            #endif
        }
        #ifdef VERBOSE
        printf("]\n");
        #endif

        if (bad) {
            fprintf(stderr, "Rejected since the 0-vector was input...\n");
            return_code = -3;
            goto end;
        }
    }

    // seed the RNG
    // ------------
    rng_state[0] = splitmix64(&opts->seed);
    rng_state[1] = splitmix64(&opts->seed);
    rng_state[2] = splitmix64(&opts->seed);
    rng_state[3] = splitmix64(&opts->seed);

    // get an initial simplex
    // ----------------------
    // shuffle the labels using Fisher-Yates
    if (opts->random)
        fisher_yates(labels, num_vecs, rng_state);

    // begin with seed_simp = labels[0],labels[1],labels[2],...
    for (int i=0; i<dim; ++i) _inds[i] = i;

    // DEBUG PRINT
    #if defined(DEBUG) || defined(VERBOSE)
        fprintf(stderr,"Constructing initial simplex...\n");
    #endif

    while (1) {
        // for retrying next iteration w/ goto
        begin_seed:;
        #ifdef DEBUG
        fprintf(stderr, "new attempt for seed simplex... ");
        #endif

        // get simplex labels, sort them
        for (int i = 0; i < dim; i++) simp_labels[i] = labels[_inds[i]];
        insertion_sort(simp_labels, dim);

        // DEBUG PRINT
        #if defined(DEBUG) || defined(VERBOSE)
            fprintf(stderr,"[");
            for (int i=0; i<dim; ++i)
                fprintf(stderr,"%d,",simp_labels[i]);
            fprintf(stderr,"]\n");
        #endif

        // get the V-representation
        for (int i=0; i<dim; ++i) {
            for (int j=0; j<dim; ++j) {
                seed_simp_V[dim* i+j] = vecs[dim* simp_labels[i]+j];
            }
        }

        // check determinant
        if (det(seed_simp_V, dim) == 0) {
            increment_inds:;

            // increment the indices corresponding to the simplex
            // iterate over _inds right-to-left, trying to increment each val
            int i = dim - 1;
            while (i >= 0 && _inds[i] >= num_vecs - dim + i) {
                // (second condition sees if we can update _inds[j] for j>i)
                i--;
            }

            // exhausted all combinations... error
            // (shouldn't ever hit though)
            if (i < 0) { return_code = -4; goto end; }

            // update the index i
            _inds[i]++;

            // update the indices j>i
            for (int j = i + 1; j < dim; j++)
                _inds[j] = _inds[i] + (j - i);
            goto begin_seed;
        }

        // get H-representation
        hrep(seed_simp_V, dim, seed_simp_H);

        // write as formal simplex
        // -----------------------
        Simplex *simp = &_simps[0];

        for (int i=0; i<dim; ++i) {
            simp->labels[i] = simp_labels[i];

            simp->external_facet_inds[i] = i; // all facets begin as external
            for (int j=0; j<dim; ++j)
                simp->normals[MAX_DIM* i+j] = seed_simp_H[dim* i+j];
        }
        simp->num_external_facets = dim;

        // check if any other vector is included in this cone
        // --------------------------------------------------
        num_contained = simp_contains(
            simp, vecs, dim, labels, num_vecs,
            opts->fine, contained_labels);

        // if we care about fineness, must repeat until we get num_contained=0
        if (opts->fine && (num_contained != 0)) {
            // darn... another vector is in cone... subdivide until we're good
            int cont_label = contained_labels[0];
            int cont_ind;

            // get contained vector index
            for (int i=0; i<num_vecs; ++i) {
                if (cont_label == labels[i]) {
                    cont_ind = i;
                    break;
                }
            }

            // try replacing ith point with cont_ind
            for (int i=0; i<dim; ++i) {
                // try replacing the point i with the interior point
                int tmp  = _inds[i];
                _inds[i] = cont_ind;

                // get the V-representation
                for (int j=0; j<dim; ++j) {
                    for (int k=0; k<dim; ++k) {
                        seed_simp_V[dim* j+k] = vecs[dim* labels[_inds[j]]+k];
                    }
                }

                // check determinant
                if (det(seed_simp_V, dim) != 0) { goto begin_seed; }

                _inds[i] = tmp;
            }
            // this line should never be hit
            fprintf(stderr, "Cone contained label %d but ", cont_label);
            fprintf(stderr, "couldn't replace any of its vertices while ");
            fprintf(stderr, "staying full-dim. Deeper error exists");
            return_code = -100;
            goto end;
        }

        break;
    }

    // discard used labels
    // -------------------
    for (int i=0; i<num_labels; ++i) {
        for (int j=0; j<dim; ++j) {
            if (labels[i] == _simps[0].labels[j]) {
                // matches a label in simp... throw it away
                labels[i] = labels[num_labels-1];
                num_labels--;
                i--; // decrement i in case swapped-in matches another label
                break;
            }
        }
    }

    for (int i=0; i<num_labels; ++i) {
        for (int j=0; j<num_contained; ++j) {
            if (labels[i] == contained_labels[j]) {
                // matches a contained label... throw it away
                labels[i] = labels[num_labels-1];
                num_labels--;
                i--; // decrement i in case swapped-in matches another label
                break;
            }
        }
    }

    // update simp count
    (*num_simps)++;

    #if defined(DEBUG) || defined(VERBOSE)
        fprintf(stderr,"Done!\n");
    #endif

    // build other simplices
    // ---------------------
    visible_isimp  = malloc(max_num_simps * dim * sizeof(int));
    visible_ifacet = malloc(max_num_simps * dim * sizeof(int));
    if (visible_isimp == NULL)  { return_code = -2; goto end; }
    if (visible_ifacet == NULL) { return_code = -2; goto end; }

    int last_num_labels = num_labels+1;
    #if defined(DEBUG) || defined(VERBOSE)
    fprintf(stderr, "\n");
    fprintf(stderr, "#remaining labels | #simps\n");
    fprintf(stderr, "--------------------------\n");
    #endif
    while (num_labels > 0) {
        // ensure we're making progress
        if (last_num_labels <= num_labels) {
            #ifdef VERBOSE
            printf("Didn't make progress in last iteration... %d %d\n",last_num_labels,num_labels);
            #endif

            return_code = -5;
            goto end;
        }
        last_num_labels = num_labels;

        // re-shuffle the (now trimmed) labels
        if (opts->random)
            fisher_yates(labels, num_labels, rng_state);

        #if defined(DEBUG) || defined(VERBOSE)
        fprintf(stderr, "%d | %d\n", num_labels, *num_simps);
        #endif

        // try pushing each label until one works (doesn't cover another vector)
        for (int ilabel=0; ilabel<num_labels; ++ilabel) {
            visible_numfacets = 0;
            int label = labels[ilabel];

            // get geometric vector associated to the label
            int v[dim];
            for (int i=0; i<dim; ++i) v[i] = vecs[dim* label+i];

            // compute visible facets
            // ----------------------
            for (int isimp=0; isimp<*num_simps; ++isimp) {
                // does the isimp-th simplex have a visible facet?
                // (need dot(v, normal)<0)
                Simplex *simp = &_simps[isimp];

                for (int ifacet=0; ifacet<simp->num_external_facets; ++ifacet) {
                    int dotted = dot(
                        &simp->normals[MAX_DIM*simp->external_facet_inds[ifacet]],
                        v,
                        dim);

                    if (dotted < 0) {
                        visible_isimp[visible_numfacets]  = isimp;
                        visible_ifacet[visible_numfacets] = ifacet;
                        visible_numfacets++;
                    }
                }
            }

            if (*num_simps + visible_numfacets > max_num_simps) {
                return_code = -6;
                goto end;
            }

            // tentatively add simps associated to visible facets
            // --------------------------------------------------
            // (don't update num_simps yet in case any of these are bad)
            for (int k=0; k<visible_numfacets; ++k) {
                Simplex *facet_haver = &_simps[visible_isimp[k]];

                // store simplex at index (*num_simps + k)
                // i.e., store k+1 indices after the 'recorded' length
                Simplex *simp = &_simps[*num_simps+k];

                // make the simplex
                // ----------------
                simp->labels[0] = label;

                // collect the labels from the external facet
                int skipped = 0;
                for (int i=0; i<dim; ++i) {
                    // ith facet corresponds to deleting ith point
                    int actual_fi = facet_haver->external_facet_inds[visible_ifacet[k]];
                    skipped = skipped || (i==actual_fi);
                    if (i==actual_fi) {
                        continue; // deleted point
                    }

                    simp->labels[(i-skipped)+1] = facet_haver->labels[i];
                }

                // get the rays/hyperplanes to check for covering other vecs
                for (int i=0; i<dim; ++i) {
                    for (int j=0; j<dim; ++j) {
                        seed_simp_V[dim* i+j] = vecs[dim* simp->labels[i]+j];
                    }
                }
                hrep(seed_simp_V, dim, seed_simp_H);

                for (int i=0; i<dim; ++i) {
                    for (int j=0; j<dim; ++j)
                        simp->normals[MAX_DIM* i+j] = seed_simp_H[dim* i+j];
                }

                // all but 0th facet (which deletes new point) are external
                for (int i=1; i<dim; ++i) {
                    simp->external_facet_inds[i-1] = i;
                }
                simp->num_external_facets = dim-1;

                // check if there is a bad containment
                // -----------------------------------
                num_contained = simp_contains(
                    simp, vecs, dim, labels, num_labels,
                    opts->fine, contained_labels);

                if (opts->fine && num_contained != 0) {
                    #ifdef DEBUG
                    fprintf(stderr,"Simplex [");
                    for (int i=0; i<dim; ++i) {
                        fprintf(stderr,"%d,",simp->labels[i]);
                    }
                    fprintf(stderr,"] contained vec %d\n", contained_labels[0]);
                    #endif

                    break;
                }
            }
            // try next label if one of the simplices covered another vec
            if (opts->fine && num_contained) {
                continue;
            }

            // not bad :)
            // ----------
            // update external facets...
            // first, remove the visible facets from being external
            for (int k=0; k<visible_numfacets; ++k) {
                Simplex *facet_haver = &_simps[visible_isimp[k]];
                int ifacet  = visible_ifacet[k];
                int last_idx = facet_haver->num_external_facets - 1;

                // swap-with-last removal
                facet_haver->external_facet_inds[ifacet] = facet_haver->external_facet_inds[last_idx];
                facet_haver->num_external_facets--;

                // fix up any later entries that pointed to last_idx on the same simplex
                for (int k2 = k+1; k2 < visible_numfacets; ++k2) {
                    if (visible_isimp[k2] == visible_isimp[k] && visible_ifacet[k2] == last_idx) {
                        visible_ifacet[k2] = ifacet;
                        // (at most one such k2 exists per k)
                    }
                }
            }

            // now, remove any new facets that are included in 2x new simps
            // (don't compare normals since diff dim-1 cones can be coplanar)
            for (int i=0; i<visible_numfacets; ++i) {
                Simplex *simpA = &_simps[*num_simps + i];

                for (int j=i+1; j<visible_numfacets; ++j) {
                    Simplex *simpB = &_simps[*num_simps + j];

                    // count shared vertices; track the unique vertex index
                    int shared_count = 0;
                    int a_unique = -1; // index in simpA of vertex not in simpB
                    int b_unique = -1; // index in simpB of vertex not in simpA
                    for (int ka=0; ka<dim; ++ka) {
                        int found = 0;
                        for (int kb=0; kb<dim; ++kb) {
                            if (simpA->labels[ka] == simpB->labels[kb]) {
                                found = 1;
                                break; }
                        }
                        if (found)
                            shared_count++;
                        else
                            a_unique = ka;
                    }
                    if (shared_count != dim-1) continue; // not a shared facet
                    for (int kb=0; kb<dim; ++kb) {
                        int found = 0;
                        for (int ka=0; ka<dim; ++ka) {
                            if (simpB->labels[kb] == simpA->labels[ka]) { found = 1; break; }
                        }
                        if (!found) { b_unique = kb; break; }
                    }

                    // remove facets a_unique and b_unique from simpA, simpB
                    for (int ifacet=0; ifacet<simpA->num_external_facets; ++ifacet) {
                        if (simpA->external_facet_inds[ifacet] == a_unique) {
                            simpA->external_facet_inds[ifacet] = simpA->external_facet_inds[simpA->num_external_facets-1];
                            simpA->num_external_facets--;
                            break;
                        }
                    }
                    // remove b_unique from simpB's external facets
                    for (int ifacet=0; ifacet<simpB->num_external_facets; ++ifacet) {
                        if (simpB->external_facet_inds[ifacet] == b_unique) {
                            simpB->external_facet_inds[ifacet] = simpB->external_facet_inds[simpB->num_external_facets-1];
                            simpB->num_external_facets--;
                            break;
                        }
                    }
                }
            }

            // update num_simps, labels
            (*num_simps) += visible_numfacets;

            for (int i=0; i<num_labels; ++i) {
                // check if we need to throw away the ith label
                if (labels[i] == label) {
                    labels[i] = labels[num_labels-1];
                    num_labels--;
                    break;
                }
            }
            for (int i=0; i<num_labels; ++i) {
                // check if we need to throw away the ith label
                for (int j=0; j<num_contained; ++j) {
                    if (labels[i] == contained_labels[j]) {
                        // matches a contained label... throw it away
                        labels[i] = labels[num_labels-1];
                        num_labels--;
                        i--; // decrement i in case swapped-in matches another label
                        break;
                    }
                }
            }
            break;
        }
    }

    // save the simplices in the output object
    for (int i=0; i<*num_simps; ++i) {
        insertion_sort(_simps[i].labels, dim);
        for (int j=0; j<dim; ++j) {
            simps[dim* i+j] = _simps[i].labels[j];
        }
    }

    // end goto
    // --------
    end:
        opts->seed = next(rng_state); // update the seed (in case multiple calls are made)

        free(_simps);
        free(visible_isimp);
        free(visible_ifacet);
        return return_code;
}

#endif

#endif
