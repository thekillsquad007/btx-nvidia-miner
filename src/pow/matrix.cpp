#include "pow/matrix.h"

#include "pow/field.h"

#include <cassert>
#include <cstdint>

namespace btx {
namespace pow {

Matrix::Matrix(uint32_t rows, uint32_t cols)
    : m_rows(rows), m_cols(cols), m_data(static_cast<size_t>(rows) * cols, 0)
{
}

Matrix::Matrix(const Matrix& other)
    : m_rows(other.m_rows), m_cols(other.m_cols), m_data(other.m_data)
{
}

Matrix::Matrix(Matrix&& other) noexcept
    : m_rows(other.m_rows), m_cols(other.m_cols), m_data(std::move(other.m_data))
{
    other.m_rows = 0;
    other.m_cols = 0;
}

Matrix& Matrix::operator=(const Matrix& other)
{
    if (this != &other) {
        m_rows = other.m_rows;
        m_cols = other.m_cols;
        m_data = other.m_data;
    }
    return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept
{
    if (this != &other) {
        m_rows = other.m_rows;
        m_cols = other.m_cols;
        m_data = std::move(other.m_data);
        other.m_rows = 0;
        other.m_cols = 0;
    }
    return *this;
}

field::Element& Matrix::at(uint32_t row, uint32_t col)
{
    assert(row < m_rows && col < m_cols);
    return m_data[static_cast<size_t>(row) * m_cols + col];
}

const field::Element& Matrix::at(uint32_t row, uint32_t col) const
{
    assert(row < m_rows && col < m_cols);
    return m_data[static_cast<size_t>(row) * m_cols + col];
}

Matrix Matrix::block(uint32_t bi, uint32_t bj, uint32_t b) const
{
    assert(b > 0);
    const uint32_t row0 = bi * b;
    const uint32_t col0 = bj * b;
    assert(row0 + b <= m_rows && col0 + b <= m_cols);
    Matrix out(b, b);
    for (uint32_t r = 0; r < b; ++r) {
        for (uint32_t c = 0; c < b; ++c) {
            out.at(r, c) = at(row0 + r, col0 + c);
        }
    }
    return out;
}

Matrix Matrix::operator+(const Matrix& rhs) const
{
    assert(m_rows == rhs.m_rows && m_cols == rhs.m_cols);
    Matrix out(m_rows, m_cols);
    for (size_t i = 0; i < m_data.size(); ++i) {
        out.m_data[i] = field::add(m_data[i], rhs.m_data[i]);
    }
    return out;
}

Matrix Matrix::operator-(const Matrix& rhs) const
{
    assert(m_rows == rhs.m_rows && m_cols == rhs.m_cols);
    Matrix out(m_rows, m_cols);
    for (size_t i = 0; i < m_data.size(); ++i) {
        out.m_data[i] = field::sub(m_data[i], rhs.m_data[i]);
    }
    return out;
}

Matrix Matrix::operator*(const Matrix& rhs) const
{
    assert(m_cols == rhs.m_rows);
    Matrix out(m_rows, rhs.m_cols);
    std::vector<field::Element> col(rhs.m_rows);
    for (uint32_t i = 0; i < m_rows; ++i) {
        const field::Element* row_ptr = &m_data[static_cast<size_t>(i) * m_cols];
        for (uint32_t j = 0; j < rhs.m_cols; ++j) {
            for (uint32_t k = 0; k < rhs.m_rows; ++k) col[k] = rhs.at(k, j);
            out.at(i, j) = field::dot(row_ptr, col.data(), m_cols);
        }
    }
    return out;
}

void Matrix::set_block(uint32_t bi, uint32_t bj, uint32_t b, const Matrix& blk)
{
    assert(blk.rows() == b && blk.cols() == b);
    const uint32_t row0 = bi * b;
    const uint32_t col0 = bj * b;
    for (uint32_t r = 0; r < b; ++r)
        for (uint32_t c = 0; c < b; ++c)
            at(row0 + r, col0 + c) = blk.at(r, c);
}

Matrix FromSeed(const uint256& seed, uint32_t n)
{
    Matrix m(n, n);
    for (uint32_t r = 0; r < n; ++r) {
        for (uint32_t c = 0; c < n; ++c) {
            m.at(r, c) = field::from_oracle(seed, r * n + c);
        }
    }
    return m;
}

} // namespace pow
} // namespace btx
