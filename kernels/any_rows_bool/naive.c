// Writes to out[r] whether any of the cols elements of row r of m is true, for each of the n / cols whole rows. Returns how many.
size_t any_rows_bool(const bool *m, bool *out, size_t n, size_t cols) {
    size_t rows = n / cols;
    for (size_t r = 0; r < rows; r++) {
        unsigned char any = 0;
        for (size_t c = 0; c < cols; c++) {
            any |= (unsigned char)m[r * cols + c];
        }
        out[r] = any != 0;
    }
    return rows;
}
