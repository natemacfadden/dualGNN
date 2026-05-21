#ifndef GROW2D_H
#define GROW2D_H

//#define VERBOSE
//#define DEBUG


// HEADER
// ======
#include <stdint.h>

/*
**Description:**
Grows a fine triangulation of a 2D lattice polygon.

Starts from a random unimodular triangle. Iteratively adds simplices by
    1) selecting an exterior (but not boundary of the polygon) edge, e, of the
       current simplicial complex (used by only 1 simplex),
    2) defining a random ordering of the points in the polygon,
    3) iterating over the points [pt0, pt1, ...], checking if the triangle
       {e, pti} could be added to the simplicial complex (intersection
       property) and whether it doesn't cover any other points (so we can get a
       fine triangulation),
    4) as soon as a point is found, add that simplex and go back to 1 .

Ported from my previous implementation here:
    cytools/ntfe/face_triangulations.py @ 1e09ca4.

**Arguments:**
- `pts`:           Input lattice points as a flat (num_pts x 2) row-major array.
- `num_pts`:       Number of input points.
- `bdry`:          Boundary edges as a flat (num_bdry x 2) row-major array,
                   each pair (u, v) with u < v.
- `num_bdry`:      Number of boundary edges.
- `seed`:          RNG seed (updated on return for chaining calls).
// output
- `max_num_simps`: Max allowed number of simplices.
- `simps`:         OUTPUT: Simplices as a flat (num_simps x 3) row-major array,
                   each triple (i, j, k) with i < j < k.
- `num_simps`:     OUTPUT: Number of simplices written.

**Returns:**
A status code:
     0: success
    -1: 0 points input
    -2: memory allocation problem
    -3: could not find an initial unit-area triangle
    -4: failed to grow (no compatible vertex found for some edge)
    -5: exceeded max_num_simps
  -100: edge already fully used (should never happen; indicates a bug)
*/

int grow2d(
    int *pts,
    int num_pts,
    int *bdry,
    int num_bdry,
    uint64_t *seed,
    int max_num_simps,
    uint32_t *simps,
    int *num_simps
);


// IMPLEMENTATION
// ==============
#ifdef GROW2D_IMPLEMENTATION

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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


// GEOMETRY HELPERS
// ----------------
static inline int min(int a, int b) { return a < b ? a : b; }
static inline int max(int a, int b) { return a > b ? a : b; }

static int triangle_area_2x(int *pts, int a, int b, int c) {
    // twice the triangle area via shoelace formula
    int64_t x0=pts[2*a], y0=pts[2*a+1];
    int64_t x1=pts[2*b], y1=pts[2*b+1];
    int64_t x2=pts[2*c], y2=pts[2*c+1];
    int64_t val = x0*(y1-y2) + x1*(y2-y0) + x2*(y0-y1);
    return (int)(val < 0 ? -val : val);
}

static int ccw(int *pts, int A, int B, int C) {
    // 1 iff A->B->C is strictly counter-clockwise
    int64_t ax=pts[2*A], ay=pts[2*A+1];
    int64_t bx=pts[2*B], by=pts[2*B+1];
    int64_t cx=pts[2*C], cy=pts[2*C+1];
    return (bx-ax)*(cy-ay) > (by-ay)*(cx-ax);
}

static int seg_intersect(int *pts, int A, int B, int C, int D) {
    // 1 iff open segments AB and CD intersect in their strict interiors
    // (returns 0 if they share an endpoint)
    if (A==C || A==D || B==C || B==D) return 0;
    return (ccw(pts,A,C,D) != ccw(pts,B,C,D))
        && (ccw(pts,A,B,C) != ccw(pts,A,B,D));
}


// EDGE BOOKKEEPING
// ----------------
typedef struct {
    int u, v;           // trust u < v always
    int choosable;      // 1 if exterior of complex but not boundary of polygon
    int min_x, min_y;   // bounding box (for intersection pre-filter)
    int max_x, max_y;
} Edge;

static int find_edge(Edge *edges, int num_edges, int u, int v) {
    // returns index of edge (u,v), or -1 if not found
    for (int i = 0; i < num_edges; i++)
        if (edges[i].u == u && edges[i].v == v) return i;
    return -1;
}

static int is_bdry(int *bdry, int num_bdry, int u, int v) {
    for (int i = 0; i < num_bdry; i++)
        if (bdry[2*i] == u && bdry[2*i+1] == v) return 1;
    return 0;
}

static void set_bbox(Edge *e, int *pts) {
    e->min_x = min(pts[2*e->u],   pts[2*e->v]);
    e->min_y = min(pts[2*e->u+1], pts[2*e->v+1]);
    e->max_x = max(pts[2*e->u],   pts[2*e->v]);
    e->max_y = max(pts[2*e->u+1], pts[2*e->v+1]);
}

static void add_edge(
    Edge *edges, int *num_edges,
    int *extendable_edges, int *num_extendable_edges,
    int *pts, int u, int v, int choosable)
{
    // add edge (u,v) to the edges array
    // append to extendable_edges if choosable=1
    Edge *e = &edges[*num_edges];
    
    e->u = u;
    e->v = v;
    
    e->choosable = choosable;
    
    set_bbox(e, pts);
    
    if (choosable)
        extendable_edges[(*num_extendable_edges)++] = *num_edges;

    (*num_edges)++;
}

static void remove_choosable(
    Edge *edges,
    int *extendable_edges, int *num_extendable_edges,
    int chosen_edge)
{
    // remove edge chosen_edge from choosable (swap-with-last)
    edges[chosen_edge].choosable = 0;
    for (int k = 0; k < *num_extendable_edges; k++) {
        if (extendable_edges[k] == chosen_edge) {
            extendable_edges[k] = extendable_edges[--(*num_extendable_edges)];
            return;
        }
    }
}


// GROW2D
// ======
int grow2d(
    int *pts,
    int num_pts,
    int *bdry,
    int num_bdry,
    uint64_t *seed,
    int max_num_simps,
    uint32_t *simps,
    int *num_simps
)
{
    // input check
    if (num_pts == 0) return -1;

    // set up variables
    // ----------------
    int return_code = 0;
    uint64_t rng_state[4];

    int num_edges            = 0;
    int num_extendable_edges = 0;
    int max_edges            = 2*max_num_simps + 3;

    Edge *edges            = NULL;
    int  *extendable_edges = NULL;
    int  *pts_to_try       = NULL;

    *num_simps = 0;

    // allocate
    edges            = malloc(max_edges * sizeof(Edge));
    extendable_edges = malloc(max_edges * sizeof(int));
    pts_to_try       = malloc(num_pts   * sizeof(int));

    if (!edges || !extendable_edges || !pts_to_try) {
        return_code = -2;
        goto end;
    }

    // seed the RNG
    // ------------
    rng_state[0] = splitmix64(seed);
    rng_state[1] = splitmix64(seed);
    rng_state[2] = splitmix64(seed);
    rng_state[3] = splitmix64(seed);

    // find initial unit-area triangle
    // --------------------------------
    int v0, v1, v2;
    {
        int max_tries = num_pts * num_pts + 100;
        for (int _t = 0; _t < max_tries; _t++) {
            v0 = next(rng_state) % num_pts;
            do { v1 = next(rng_state) % num_pts; } while (v1 == v0);
            do { v2 = next(rng_state) % num_pts; } while (v2 == v0 || v2 == v1);
            if (triangle_area_2x(pts, v0, v1, v2) == 1) goto found_seed;
        }
        return_code = -3; goto end;
        found_seed:;
    }

    // sort v0 <= v1 <= v2
    if (v0 > v1) { int t=v0; v0=v1; v1=t; }
    if (v1 > v2) { int t=v1; v1=v2; v2=t; }
    if (v0 > v1) { int t=v0; v0=v1; v1=t; }

    #if defined(DEBUG) || defined(VERBOSE)
    fprintf(stderr, "Seed simplex: [%d, %d, %d]\\n", v0, v1, v2);
    #endif

    // record seed simplex
    simps[0] = v0;
    simps[1] = v1;
    simps[2] = v2;
    (*num_simps)++;

    // add 3 edges (sorted pairs); mark choosable if not boundary
    {
        int edge_pairs[3][2] = {{v0,v1},{v0,v2},{v1,v2}};
        for (int k = 0; k < 3; k++) {
            int u=edge_pairs[k][0], v=edge_pairs[k][1];
            int choosable = !is_bdry(bdry, num_bdry, u, v);
            add_edge(edges, &num_edges, extendable_edges, &num_extendable_edges,
                     pts, u, v, choosable);
        }
    }

    #if defined(DEBUG) || defined(VERBOSE)
    fprintf(stderr, "\n#choosable | #simps\n");
    fprintf(stderr,   "------------------\n");
    #endif

    // grow
    // ----
    while (num_extendable_edges > 0) {
        #if defined(DEBUG) || defined(VERBOSE)
        fprintf(stderr, "%d | %d\n", num_extendable_edges, *num_simps);
        #endif

        // pick a random choosable edge
        int chosen_edge_i = next(rng_state) % num_extendable_edges;
        int chosen_edge = extendable_edges[chosen_edge_i];
        int u = edges[chosen_edge].u, v = edges[chosen_edge].v;

        #ifdef VERBOSE
        fprintf(stderr, "Building off edge (%d,%d)...\n", u, v);
        #endif

        // build and shuffle candidate vertices
        int pts_to_try_n = 0;
        for (int i = 0; i < num_pts; i++)
            if (i != u && i != v)
                pts_to_try[pts_to_try_n++] = i;
        fisher_yates(pts_to_try, pts_to_try_n, rng_state);

        int found = 0;
        for (int ti = 0; ti < pts_to_try_n; ti++) {
            int vi = pts_to_try[ti];

            // unit-area check (Pick's theorem)
            if (triangle_area_2x(pts, u, v, vi) != 1) continue;

            // new edges (sorted pairs)
            int e0u = min(u,vi), e0v = max(u,vi);
            int e1u = min(v,vi), e1v = max(v,vi);

            int e0_idx = find_edge(edges, num_edges, e0u, e0v);
            int e1_idx = find_edge(edges, num_edges, e1u, e1v);

            // skip if this triangle is already in simps
            // (check if both edges exist. If so, scan over simplices to search
            //  for matches)
            if (e0_idx >= 0 && e1_idx >= 0) {
                int a=u, b=v, c=vi;

                if (a>b){int t=a;a=b;b=t;}
                if (b>c){int t=b;b=c;c=t;}
                if (a>b){int t=a;a=b;b=t;}

                int dup = 0;
                for (int k = 0; k < *num_simps; k++) {
                    if ((int)simps[3*k]   == a &&
                        (int)simps[3*k+1] == b &&
                        (int)simps[3*k+2] == c)
                        { dup=1; break; }
                }
                if (dup) continue;
            }

            // intersection check for each genuinely new edge
            for (int i = 0; i < 2; i++) {
                if ((i == 0) & (e0_idx >= 0)) continue;
                if ((i == 1) & (e1_idx >= 0)) continue;

                Edge new_edge;
                if (i == 0) { new_edge.u = e0u; new_edge.v = e0v; }
                else        { new_edge.u = e1u; new_edge.v = e1v; }
                set_bbox(&new_edge, pts);

                for (int k = 0; k < num_edges; k++) {
                    // bbox pre-filter
                    if (edges[k].max_x < new_edge.min_x ||
                        new_edge.max_x < edges[k].min_x ||
                        edges[k].max_y < new_edge.min_y ||
                        new_edge.max_y < edges[k].min_y)
                        continue;

                    if (seg_intersect(pts, new_edge.u, new_edge.v,
                                      edges[k].u, edges[k].v))
                        goto try_next_vertex;
                }
            }

            // accepted
            // --------
            #ifdef VERBOSE
            fprintf(stderr, "  Accepted vertex %d\n", vi);
            #endif

            if (*num_simps >= max_num_simps) { return_code = -5; goto end; }

            // record simplex (sorted)
            {
                int a=u, b=v, c=vi;
                
                if (a>b){int t=a;a=b;b=t;}
                if (b>c){int t=b;b=c;c=t;}
                if (a>b){int t=a;a=b;b=t;}
                
                simps[3*(*num_simps)]   = a;
                simps[3*(*num_simps)+1] = b;
                simps[3*(*num_simps)+2] = c;
                (*num_simps)++;
            }

            // remove current edge from choosable
            edges[chosen_edge].choosable = 0;
            extendable_edges[chosen_edge_i] =
                extendable_edges[--num_extendable_edges];

            // add or update the two new edges
            for (int i = 0; i < 2; i++) {
                int new_u   = (i == 0) ? e0u    : e1u;
                int new_v   = (i == 0) ? e0v    : e1v;
                int new_idx = (i == 0) ? e0_idx : e1_idx;
                int on_bdry = is_bdry(bdry, num_bdry, new_u, new_v);

                if (new_idx >= 0) {
                    if (!on_bdry && edges[new_idx].choosable) {
                        // interior edge now shared by 2 triangles -> remove
                        remove_choosable(edges, extendable_edges,
                                         &num_extendable_edges, new_idx);
                    } else {
                        // edge already fully used -- should never happen
                        return_code = -100; goto end;
                    }
                } else {
                    // new edge
                    if (num_edges >= max_edges) { return_code = -5; goto end; }

                    add_edge(edges, &num_edges, extendable_edges,
                             &num_extendable_edges, pts,
                             new_u, new_v, !on_bdry);
                }
            }

            found = 1;
            break;
            try_next_vertex:;
        }

        if (!found) {
            #ifdef VERBOSE
            fprintf(stderr,
                "Failed: no compatible vertex for edge (%d,%d)\n",
                u, v);
            #endif
            return_code = -4;
            goto end;
        }
    }

    // end goto
    // --------
    end:
        *seed = next(rng_state);

        free(edges);
        free(extendable_edges);
        free(pts_to_try);
        return return_code;
}

#endif // GROW2D_IMPLEMENTATION

#endif // GROW2D_H
