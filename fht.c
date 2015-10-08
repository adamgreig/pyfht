#include <stdio.h>
#include <stdint.h>
#include <strings.h>

/*
 * Fast Hadamard Transform
 * N: size of transform, MUST be a power of 2 or this will segfault
 * u: N-long array to be transformed in-place
 */
void fht(uint32_t N, double* u)
{
    double temp;
    uint32_t i, j;

    for(i=N>>1; i>0; i>>=1) {
        for(j=0; j<N; j++) {
            if((i & j) == 0) {
                temp   = u[j];
                u[j]  += u[i|j];
                u[i|j] = temp - u[i|j];
            }
        }
    }
}

/* Taps for maximal-length LFSRs of 0 to 32 bits length.
 * These particular taps are the largest taps from:
 * http://users.ece.cmu.edu/~koopman/lfsr/index.html
 * that have the maximal number of bits set (i.e., the largest number of taps)
 * for any set in that size of LFSR.
 * This special property ensures we get fewer runs of decreasing powers of two
 * caused by single bits being mostly unchanged as they traverse large runs
 * without taps, which selects for rows of the Hadamard matrix which have
 * powers-of-two indices, which tend to be highly structured with relation to
 * each other.
 *
 * From the compressed files on the site above, we can find the taps to use for
 * all n>4 (for n<4, use 0, 0, 3, 6):
 *
 * rustc -O findmaxtaps.rs
 * for f in *.dat.gz;
 *    gunzip -c $f | findmaxtaps;
 * end
 *
 *
 * Why use LFSRs?
 * 
 * We want all numbers between 1 and N (where N is the power-of-two size of the
 * Hadamard matrix we're using) to be equally likely, in a random sequence with
 * no repeats (until the entire sequence repeats). We specifically don't want 0
 * (as the 0th row of a Hadamard matrix is all-1s and a very poor choice
 * always) and we cannot afford repeats (we must select a subset of the rows
 * that has cardinality n).
 * 
 * With a normal random number generator we get random numbers with repeats
 * over a huge range. So we'd need to exclude 0s and keep track of which
 * numbers we've already seen, plus it's slower.
 *
 * A suitable LFSR will generate every number of that bit width exactly once
 * before repeating, and will never generate 0 (unless it starts at 0). This
 * makes them very suited to this application. However if we use a minimal
 * number of taps (the usual desire), we obtain sequences where a single 1 is
 * shifted through the register without being flipped, causing sequences of
 * powers of two: 128->64->32->16->8 etc. This is a particularly poor choice of
 * sampling from the Hadamard matrix (it results in a new matrix which still
 * has high levels of structure) so we deliberately pick taps with as many bits
 * set as possible, to ensure as much bit flipping as possible.
 */
uint32_t TAPS[33] = {
    0x0, 0x0, 0x3, 0x6, 0xC, 0x1E, 0x39, 0x7E, 0xFA, 0x1FD, 0x3FC, 0x7F4,
    0xFDE, 0x1FFE, 0x3FF3, 0x7FFE, 0xFFF6, 0x1FFF7, 0x3FFF3, 0x7FFE9, 0xFFFFC,
    0x1FFFD9, 0x3FFFF6, 0x7FFFFE, 0xFFFFD7, 0x1FFFFF7, 0x3FFFFDD, 0x7FFFFF1,
    0xFFFFBFB, 0x1FFFFFDF, 0x3FFFFF6F, 0x7FFFFFFB, 0xFFFFFFFA};

/*
 * Shuffle a (of length N) into b (of length n) in a random order.
 * N MUST be a power of 2
 * n may be any integer n < M
 * lfsr is a seed, must 1<=lfsr<N
 * NB a[0] will never be used
 */
void shuffle_smaller(uint32_t N, double* a, uint32_t n, double* b, uint32_t lfsr)
{
    uint32_t i;
    uint8_t z = ffs(N) - 1;
    uint32_t tap = TAPS[z];

    for(i=0; i<n; i++) {
        if(lfsr & 1)
            lfsr = (lfsr >> 1) ^ tap;
        else
            lfsr >>= 1;
        b[i] = a[lfsr];
    }
}

/*
 * Shuffle b (of length n) into a (of length N) in a random order,
 * not changing the other elements of a.
 * N MUST be a power of 2
 * n may be any integer n < M
 * lfsr is a seed, must 1<=lfsr<N
 * NB a[0] will never be assigned
 */
void shuffle_bigger(uint32_t n, double* b, uint32_t N, double* a, uint32_t lfsr)
{
    uint32_t i;
    uint8_t z = ffs(N) - 1;

    for(i=0; i<n; i++) {
        if(lfsr & 1)
            lfsr = (lfsr >> 1) ^ TAPS[z];
        else
            lfsr >>= 1;
        a[lfsr] = b[i];
    }
}
