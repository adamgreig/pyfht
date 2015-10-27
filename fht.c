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
static const uint32_t LIBFHT_TAPS[33] = {
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
void shuffle_smaller_lfsr(uint32_t N, double* a, uint32_t n, double* b, uint32_t lfsr)
{
    uint32_t i, j;
    uint8_t z = ffs(N) - 1;
    uint32_t tap = LIBFHT_TAPS[z];

    for(i=0; i<n; i++) {
        for(j=0; j<z; j++) {
            if(lfsr & 1)
                lfsr = (lfsr >> 1) ^ tap;
            else
                lfsr >>= 1;
        }
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
void shuffle_bigger_lfsr(uint32_t n, double* b, uint32_t N, double* a, uint32_t lfsr)
{
    uint32_t i, j;
    uint8_t z = ffs(N) - 1;
    uint32_t tap = LIBFHT_TAPS[z];

    for(i=0; i<n; i++) {
        for(j=0; j<z; j++) {
            if(lfsr & 1)
                lfsr = (lfsr >> 1) ^ tap;
            else
                lfsr >>= 1;
        }
        a[lfsr] = b[i];
    }
}


/*
 * Alternative approach to shuffling that is somewhat more sophisticated but
 * hopefully yields better results and allows for a greater diversity of
 * identically-sized transforms.
 *
 * The basic gist is to use an XorShift128+ generator instead of an LFSR, and
 * then use the Floyd algorithm to generate a random set of indicies. Since
 * this requires random numbers in an arbitrary [0, n) range, use a rejection
 * sampling approach to remove modulus-induced bias.
 *
 * It's upsetting that LFSRs are such a perfect match for this in all ways
 * except random quality.
 */

/* XorShift128+ from http://xorshift.di.unimi.it/xorshift128plus.c */
static uint64_t xorshift128p(uint64_t s[])
{
    uint64_t s1 = s[0];
    uint64_t s0 = s[1];
    s[0] = s0;
    s1 ^= s1 << 23;
    return (s[1] = (s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26))) + s0;
}

/* Run the XorShift128+ generator, discarding any values sufficiently small,
 * so as to remove bias induced by performing modulo with a bound not a
 * multiple of the generator maximum.
 * NB for later: xorshift128p will return numbers [1, 2^64), i.e. never 0.
 * This thresholding business assumes 0 might be returned. Need to check
 * what bias this introduces. Might be able to just subtract 1 from th.
 */
static uint64_t xsrand(uint64_t j, uint64_t s[])
{
    uint64_t r, th = -j % j;
    while((r = xorshift128p(s)) < th);
    return r % j;
}

static uint64_t floyd(uint64_t i, uint64_t n, uint64_t N,
                      uint8_t used[], uint64_t s[])
{
    uint64_t j = i + 1 + N - n;
    uint64_t r = xsrand(j, s);
    if(used[r])
        r = j;
    used[r] = 1;
    return r;
}

/*
 * Shuffle a (of length N) into b (of length n) in a random order.
 * N MUST be a power of 2
 * n may be any integer n < M
 * a[0] will never be used.
 *
 * used[] is an n-long array set to all 0s when this function is called.
 * seed0 and seed1 are unitformly distributed 64-bit numbers used as seeds.
 */
void shuffle_smaller_xs(uint32_t N, double* a, uint32_t n, double* b,
                        uint8_t used[], uint64_t seed0, uint64_t seed1)
{
    uint64_t i, r;
    uint64_t s[2] = {seed0, seed1};

    for(i=0; i<n; i++) {
        r = floyd(i, n, N, used, s);
        b[i] = a[r];
    }
}

/*
 * Shuffle b (of length n) into a (of length N) in a random order,
 * not changing the other elements of a.
 * N MUST be a power of 2
 * n may be any integer n < M
 * a[0] will never be assigned
 *
 * used[] is an n-long array set to all 0s when this function is called.
 * seed0 and seed1 are unitformly distributed 64-bit numbers used as seeds.
 */
void shuffle_bigger_xs(uint32_t n, double* b, uint32_t N, double* a,
                       uint8_t used[], uint64_t seed0, uint64_t seed1)
{
    uint64_t i, r;
    uint64_t s[2] = {seed0, seed1};

    for(i=0; i<n; i++) {
        r = floyd(i, n, N, used, s);
        a[r] = b[i];
    }
}


/*
 * One other technique. Just accept a list with the required order. Obvious.
 */


/*
 * Shuffle a (of length N) into b (of length n) in a specified order.
 * N MUST be a power of 2
 * n may be any integer n < M
 * order[] is an n-long list of the order in which to insert, don't put 0 in.
 */
void shuffle_smaller_o(uint32_t N, double* a, uint32_t n, double* b,
                       uint32_t order[])
{
    uint64_t i;
    for(i=0; i<n; i++) {
        b[i] = a[order[i]];
    }
}

/*
 * Shuffle b (of length n) into a (of length N) in a specified order,
 * not changing the other elements of a.
 * N MUST be a power of 2
 * n may be any integer n < M
 * order[] is an n-long list of the order in which to insert, don't put 0 in.
 */
void shuffle_bigger_o(uint32_t n, double* b, uint32_t N, double* a,
                      uint32_t order[])
{
    uint64_t i;
    for(i=0; i<n; i++) {
        a[order[i]] = b[i];
    }
}

