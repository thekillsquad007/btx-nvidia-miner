#pragma once

#include "pow/field.h"

#include <cstdint>
#include <vector>

class uint256;

namespace btx {
namespace pow {

class Matrix {
public:
    Matrix() : m_rows(0), m_cols(0) {}
    Matrix(uint32_t rows, uint32_t cols);
    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;
    ~Matrix() = default;

    field::Element& at(uint32_t row, uint32_t col);
    const field::Element& at(uint32_t row, uint32_t col) const;

    uint32_t rows() const { return m_rows; }
    uint32_t cols() const { return m_cols; }

    field::Element* data() { return m_data.data(); }
    const field::Element* data() const { return m_data.data(); }

    Matrix block(uint32_t bi, uint32_t bj, uint32_t b) const;
    void set_block(uint32_t bi, uint32_t bj, uint32_t b, const Matrix& blk);

    Matrix operator+(const Matrix& rhs) const;
    Matrix operator-(const Matrix& rhs) const;
    Matrix operator*(const Matrix& rhs) const;

private:
    uint32_t m_rows;
    uint32_t m_cols;
    std::vector<field::Element> m_data;
};

Matrix FromSeed(const uint256& seed, uint32_t n);

} // namespace pow
} // namespace btx
