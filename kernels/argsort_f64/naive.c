// Stable argsort of float64 keys: the indices that sort x, equal elements in
// their original order, as np.argsort(x, kind="stable") returns them.
//
// out has room for 3n elements. The result is out[0..n); the other 2n are
// working memory and hold leftovers afterwards. Returns n.
//
// A least-significant-digit radix sort. Each element becomes an unsigned
// 64-bit key whose order is NumPy's order of the elements (sort_key below),
// and the keys are sorted one digit at a time from the lowest digit up, each
// pass a counting sort, which is stable. A stable sort has exactly one right
// answer, so this returns what NumPy's timsort returns.
//
// What makes it fast is how little each pass moves:
//
// - Only the bits that differ between keys are sorted on: the window from
//   the lowest such bit to the highest. Small integers need one or two passes.
// - An element travels as one 64-bit word: the key bits not yet sorted on,
//   above its index. The index of one of n elements takes b = bits(n - 1)
//   bits, which leaves 64 - b for the key. A window wider than that must shed
//   its lowest `shed` bits before an element fits in a word, and bits are
//   shed by sorting on them: the first pass takes them as its digit (up to
//   WIDE_BITS of them) and writes packed words. When more must go, the first
//   pass writes (key, index) pairs and the second sorts on the rest of the
//   shed bits and packs.
// - The counts of every pass are taken in one pass over x, and a pass in
//   which every key has the same digit is skipped.
// - The last pass writes the indices alone, straight into out[0..n). Passes
//   alternate between two buffers, and the first is chosen by the parity of
//   the number of passes so that the last one lands in the result.
//
// The counts are on the stack: sizeof(radix_counts), 112 KB. The kernel is a
// leaf, and the smallest default thread stack on the platforms the package
// supports is 512 KB (a secondary thread on macOS).
//
// n must be below 2^26: beyond it the two pair passes would need digits
// wider than WIDE_BITS.

// Digit width of a pass over packed words.
#define DIGIT_BITS 11
// Digit width below SMALL_N elements, where clearing 2^11 counts per pass
// would cost more than the passes it saves.
#define SMALL_DIGIT_BITS 8
#define SMALL_N 2048
// Widest digit of a pass that sheds key bits. Wider digits scatter to more
// places than the first-level cache follows; 13 bits was the widest that
// still beat a pass over pairs (Apple M1 Pro).
#define WIDE_BITS 13
// Most passes any plan has: eight digits of SMALL_DIGIT_BITS cover 64 bits.
#define MAX_PASSES 8
// At or below this many elements the keys are sorted by insertion.
#define INSERTION_N 16

// The counts of every pass. The first two passes may have wide digits.
typedef struct {
    uint32_t wide[2][1 << WIDE_BITS];
    uint32_t narrow[MAX_PASSES - 2][1 << DIGIT_BITS];
} radix_counts;

// Bits needed to write v: 0 for 0.
static inline int bit_width(uint64_t v) {
    return v ? 64 - __builtin_clzll(v) : 0;
}

// The key of an element. NumPy orders float64 by value with NaN last, and
// calls two elements equal when neither is less: -0.0 and +0.0 are equal, and
// so are any two NaNs, whatever their signs and payloads. Equal elements must
// get equal keys, or the sort would order what NumPy leaves in place.
//
// A double is a sign and a magnitude, and the magnitudes order like the
// values they stand for. So a positive number's key is its magnitude above
// the middle of the range, 2^63 + m, and a negative number's is as far below
// it, 2^63 - m. Both zeros have m = 0 and get the same key. A NaN, whose
// magnitude is above infinity's, gets the largest key there is.
//
// Subtracting, where inverting every bit of a negative number would also
// order them, keeps the low bits of a key zero when the low bits of the
// element are: whole numbers of both signs then differ only in their high
// bits, and take two or three passes, where inverted keys differ in all 64
// (Apple M1 Pro: 15.2 ms against 4.7 ms for a million whole numbers from -1000 to 1000).
static inline uint64_t sort_key(double v) {
    uint64_t bits;
    __builtin_memcpy(&bits, &v, sizeof bits);
    const uint64_t sign = 0x8000000000000000ull;
    // Magnitudes above this are NaN: an all-ones exponent, a non-zero fraction.
    const uint64_t infinity = 0x7ff0000000000000ull;
    const uint64_t magnitude = bits & ~sign;
    if (magnitude > infinity) return ~0ull;
    return (bits & sign) ? sign - magnitude : sign + magnitude;
}

// Add one, for every element, to the count of its digit in each of the
// first `passes` passes. The loop is written out per number of passes so that
// the compiler keeps every shift, mask and table in a register.
#define COUNT(j) h[j][(kw >> o[j]) & m[j]]++;
#define COUNT_LOOP(body)                                   \
    for (size_t i = 0; i < n; i++) {                       \
        const uint64_t kw = sort_key(x[i]) >> lo;          \
        body                                               \
    }                                                      \
    break;

size_t argsort_f64(const double *x, int64_t *out, size_t n) {
    // The result, and the two halves of the working memory.
    uint64_t *r = (uint64_t *)out, *s1 = r + n, *s2 = s1 + n;

    if (n <= INSERTION_N) {
        // Keys go to s1, kept in order beside their indices.
        for (size_t i = 0; i < n; i++) {
            const uint64_t k = sort_key(x[i]);
            size_t j = i;
            for (; j > 0 && s1[j - 1] > k; j--) {
                s1[j] = s1[j - 1];
                out[j] = out[j - 1];
            }
            s1[j] = k;
            out[j] = (int64_t)i;
        }
        return n;
    }

    // The window: key bits [lo, lo + w) are the ones that differ somewhere.
    uint64_t any = 0, all = ~0ull;
    for (size_t i = 0; i < n; i++) {
        const uint64_t k = sort_key(x[i]);
        any |= k;
        all &= k;
    }
    const uint64_t differ = any ^ all;
    if (!differ) {
        // Every key is the same: the original order is the answer.
        for (size_t i = 0; i < n; i++) out[i] = (int64_t)i;
        return n;
    }
    const int lo = __builtin_ctzll(differ), w = bit_width(differ) - lo;
    // Bits an index takes in a packed word.
    const int b = bit_width(n - 1);
    // Low window bits that must be sorted on before a key fits above an index.
    int shed = w + b - 64;
    if (shed < 0) shed = 0;

    // The plan: bits[j] is the width of pass j's digit and off[j] its
    // position in the window, lowest digit first.
    const int target = n < SMALL_N ? SMALL_DIGIT_BITS : DIGIT_BITS;
    int bits[MAX_PASSES], off[MAX_PASSES], passes = 0, used = 0;
    // True when the first two passes go through (key, index) pairs.
    const int pairs_needed = shed > WIDE_BITS;
    if (pairs_needed) {
        bits[0] = (shed + 1) / 2;
        bits[1] = shed / 2;
        passes = 2;
        used = shed;
    } else if (shed > 0) {
        // One pass sheds them all. Make it as wide as an even split of the
        // window would, or wider if more bits than that must go.
        const int even = (w + target - 1) / target;
        bits[0] = (w + even - 1) / even;
        if (bits[0] < shed) bits[0] = shed;
        passes = 1;
        used = bits[0];
    }
    for (int rest = w - used; rest > 0;) {
        const int left = (rest + target - 1) / target, d = (rest + left - 1) / left;
        bits[passes++] = d;
        rest -= d;
    }
    off[0] = 0;
    for (int j = 1; j < passes; j++) off[j] = off[j - 1] + bits[j - 1];

    radix_counts counts;
    uint32_t *h[MAX_PASSES];
    uint64_t m[MAX_PASSES];
    int o[MAX_PASSES];
    for (int j = 0; j < passes; j++) {
        h[j] = j < 2 ? counts.wide[j] : counts.narrow[j - 2];
        m[j] = (1ull << bits[j]) - 1;
        o[j] = off[j];
        for (size_t v = 0; v <= m[j]; v++) h[j][v] = 0;
    }
    switch (passes) {
    case 1: COUNT_LOOP(COUNT(0))
    case 2: COUNT_LOOP(COUNT(0) COUNT(1))
    case 3: COUNT_LOOP(COUNT(0) COUNT(1) COUNT(2))
    case 4: COUNT_LOOP(COUNT(0) COUNT(1) COUNT(2) COUNT(3))
    case 5: COUNT_LOOP(COUNT(0) COUNT(1) COUNT(2) COUNT(3) COUNT(4))
    case 6: COUNT_LOOP(COUNT(0) COUNT(1) COUNT(2) COUNT(3) COUNT(4) COUNT(5))
    case 7: COUNT_LOOP(COUNT(0) COUNT(1) COUNT(2) COUNT(3) COUNT(4) COUNT(5) COUNT(6))
    default: COUNT_LOOP(COUNT(0) COUNT(1) COUNT(2) COUNT(3) COUNT(4) COUNT(5) COUNT(6) COUNT(7))
    }

    // Keep the passes that move something, and turn their counts into the
    // position each digit value starts at. A pass that sheds bits is kept
    // whatever its counts: the packing depends on it.
    const uint64_t first_kw = sort_key(x[0]) >> lo;
    int live[MAX_PASSES], nlive = 0;
    for (int j = 0; j < passes; j++) {
        const int sheds = pairs_needed ? j < 2 : (j == 0 && shed > 0);
        if (h[j][(first_kw >> o[j]) & m[j]] == n && !sheds) continue;
        live[nlive++] = j;
        uint32_t start = 0;
        for (size_t v = 0; v <= m[j]; v++) {
            const uint32_t c = h[j][v];
            h[j][v] = start;
            start += c;
        }
    }
    if (nlive == 0) {
        // Cannot happen while some bit of the window differs; kept so that
        // the code below never reads live[] unset.
        for (size_t i = 0; i < n; i++) out[i] = (int64_t)i;
        return n;
    }

    const uint64_t index_mask = (1ull << b) - 1;
    // Low window bits a packed word leaves out: the ones shed.
    const int dropped = shed > 0 ? used : 0;
    // The packed words as the next pass reads them, and where it writes.
    uint64_t *src, *dst;
    // The next live pass.
    int q;

    if (pairs_needed) {
        // Passes after the two that go through pairs. The pairs take 2n
        // words, and the packed words then alternate between the remaining n
        // and the first n the pairs vacate, so the pairs are put where the
        // last pass still lands in r.
        const int packed_left = nlive - 2;
        uint64_t *pairs = (packed_left & 1) ? r : s1;
        uint64_t *packed = (packed_left & 1) ? s2 : r;
        uint32_t *c0 = h[0];
        const uint64_t m0 = m[0];
        for (size_t i = 0; i < n; i++) {
            const uint64_t kw = sort_key(x[i]) >> lo;
            uint64_t *d = pairs + 2 * (size_t)c0[kw & m0]++;
            d[0] = kw;
            d[1] = i;
        }
        uint32_t *c1 = h[1];
        const uint64_t m1 = m[1];
        const int sh1 = o[1];
        if (packed_left == 0) {
            // Nothing above the shed bits differs: this is the last pass.
            for (size_t i = 0; i < n; i++) r[c1[(pairs[2 * i] >> sh1) & m1]++] = pairs[2 * i + 1];
            return n;
        }
        for (size_t i = 0; i < n; i++) {
            const uint64_t kw = pairs[2 * i];
            packed[c1[(kw >> sh1) & m1]++] = ((kw >> dropped) << b) | pairs[2 * i + 1];
        }
        src = packed;
        dst = (packed_left & 1) ? r : s1;
        q = 2;
    } else {
        const int j = live[0];
        uint32_t *c = h[j];
        const uint64_t mj = m[j];
        const int sh = o[j];
        if (nlive == 1) {
            for (size_t i = 0; i < n; i++) {
                const uint64_t kw = sort_key(x[i]) >> lo;
                r[c[(kw >> sh) & mj]++] = i;
            }
            return n;
        }
        // An odd number of passes starts in r and an even number in s1, so
        // that the last one writes r from the other buffer.
        uint64_t *first = (nlive & 1) ? r : s1;
        for (size_t i = 0; i < n; i++) {
            const uint64_t kw = sort_key(x[i]) >> lo;
            first[c[(kw >> sh) & mj]++] = ((kw >> dropped) << b) | i;
        }
        src = first;
        dst = (nlive & 1) ? s1 : r;
        q = 1;
    }

    // The passes over packed words. A digit of the window at off[j] sits at
    // b + off[j] - dropped in a word.
    for (; q < nlive - 1; q++) {
        const int j = live[q];
        uint32_t *c = h[j];
        const uint64_t mj = m[j];
        const int sh = b + o[j] - dropped;
        for (size_t i = 0; i < n; i++) {
            const uint64_t v = src[i];
            dst[c[(v >> sh) & mj]++] = v;
        }
        uint64_t *swap = src;
        src = dst;
        dst = swap;
    }
    {
        // The last pass keeps the index and drops the key.
        const int j = live[q];
        uint32_t *c = h[j];
        const uint64_t mj = m[j];
        const int sh = b + o[j] - dropped;
        for (size_t i = 0; i < n; i++) {
            const uint64_t v = src[i];
            r[c[(v >> sh) & mj]++] = v & index_mask;
        }
    }
    return n;
}
