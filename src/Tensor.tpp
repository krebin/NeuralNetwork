//
// Created by krebin on 6/15/21.
//

#include "Tensor.h"

#include <assert.h>
#include <cstring>
#include <algorithm>
#include "math.h"

/**
 * Utility function to fill an array as [start, ..., end - 1]
 *
 * @param start starting value
 * @param end ending value + 1
 * @param out array to store the range
 *
 */
template<class T>
void range(int start, int end, T *out)
{
    for (int i = start; i < end; i++)
        *(out++) = i;
}

/**
 * Tranpose of a matrix
 *
 * @param rows number of rows in input matrix
 * @param end number of colums in input matrix
 * @param in 1D representation of input matrix
 * @param out output array to store 1D representation of tranposed matrix
 *
 */
template<class T>
void _transpose(int rows, int cols, T* in, T* out)
{
    // Write transposed values into out
    for (int i = 0; i < cols; i++)
        for (int j = 0; j < rows; j++)
        {
            T val;
            int out_idx;

            val = in[i + (cols * j)];
            out_idx = j + (rows * i);
            out[out_idx] = val;
        }
}

/**
 * Matrix multiplication
 * Calculates C = AB
 * All matrices are represented as an a 1-dim array and accessed by passed in row/col parameters
 *
 * @param *A T array of length rows_A * cols_A
 * @param *B T array of length rows_B * cols_B
 * @param *C T array of length rows_A * cols_B, array will be used as output
 * @param rows_A number of rows in A
 * @param cols_A number of columns in A or rows in B
 * @param cols_B number of columns in B
 *
 */
template<class T>
void mm(T *A, T *B, T *C, int rows_A, int cols_A, int cols_B)
{
    for (int i = 0; i < cols_A; i++)
        for (int j = 0; j < rows_A; j++)
            for (int k = 0; k < cols_B; k++)
            {
                T out_val = A[i + (cols_A * j)] * B[k + (cols_B * i)];
                int out_idx = k + (cols_B * j);
                C[out_idx] += out_val;
            }
}

/**
 * Utility function to copy array into another array on heap
 * Will matrix multiply every matrix in tensor A with single matrix in tensor B
 *
 * @param source T array on stack or heap
 * @param size length of array
 *
 * @return values new copied array on heap
 *
 */
template<class T>
T* allocate_array(T* source, int size)
{
    auto values = new T[size];
    memcpy(values, source, size * sizeof(T));
    return values;
}

template<class T>
Tensor<T> bmm(const Tensor<T> &tensor_A, const Tensor<T> &tensor_B)
{
    int batch_size, C_size, rows_A, cols_A, cols_B, A_size, tensor_C_size;
    std::vector<int> new_dims;

    // Number of matrices in tensor A
    batch_size = tensor_A._dims[0];

    // Get needed row/col size for correct access of matrices represented as 1D arrays
    rows_A = tensor_A._dims[1];
    cols_A = tensor_A._dims[2];
    cols_B = tensor_B._dims[2];

    // Size of a single matrix in tensor A, B is already single matrix
    A_size = tensor_A._size / batch_size;
    auto B = tensor_B._values;

    // Size and dims of new tensor C
    C_size = rows_A * cols_B;
    tensor_C_size = batch_size * C_size;
    new_dims = {batch_size, rows_A, cols_B};

    T tensor_C_values[tensor_C_size] = {0};
    for (int i = 0; i < batch_size; i++)
    {
        // Get matrix by offsetting inner representation
        auto A = tensor_A._values + (A_size * i);

        // Make temporary array to store C = AB
        T C[C_size] = {0};
        mm(A, B, C, rows_A, cols_A, cols_B);

        // Copy matrix C into offset value for tensor_C_values
        memcpy(tensor_C_values + (C_size * i), C, sizeof(T) * C_size);
    }

    return Tensor<T>(tensor_C_values, new_dims, tensor_C_size);
}

template<class T>
Tensor<T>::Tensor(T* vals, const std::vector<int>& dims, int size)
{
    this->_dims = dims;
    this->device = "cpu";

    if (size == -1)
    {
        size = 1;

        for(int num : dims)
            size *= num;
    }

    this->_size = size;
    this->_values = allocate_array(vals, this->_size);
}

template<class T>
Tensor<T>::Tensor(const Tensor<T> &tensor) : _size(tensor._size),
                                             _dims(tensor._dims)
{
     // Copy values from input tensor and allocate new array
    this->_values = allocate_array(tensor._values, this->_size);
}

template<class T>
Tensor<T> Tensor<T>::reshape(std::vector<int> dims)
{
    // New size must be same as old size
    int size = 1;

    for(int num : dims)
        size *= num;

    assert (size == this->_size);

    // Copy Tensor and replace dims
    auto new_tensor = Tensor<T>(*this);
    new_tensor._dims = dims;

    return new_tensor;
}

template<class T>
std::ostream& operator<<(std::ostream& out, const Tensor<T>& tensor)
{
    auto dims = tensor._dims;
    int size = tensor._size;
    T *vals = tensor._values;
    std::string vals_fmt = "";

    std::vector<int> line_breaks;

    // std::cout << dims;

    if (dims.size() == 2)
    {
        line_breaks.push_back(dims[1]);
        line_breaks.push_back(dims[1]);
    }
    else if (dims.size() == 3)
    {
        line_breaks.push_back(dims[2]);
        line_breaks.push_back(dims[1] * dims[2]);
    }

    for (int i = 0; i < size; i++)
    {
        T val = *(vals + i);
        vals_fmt += std::to_string(val) + " ";

        for (auto itr = line_breaks.begin(); itr < line_breaks.end(); itr++)
        {
            if ((i + 1) % *itr == 0)
            {
                vals_fmt += "\n";
            }
        }
    }

    out << vals_fmt;

    return out;
}

template<class T>
std::string Tensor<T>::shape()
{
    std::string shape_str;

    for (auto itr = this->_dims.begin(); itr < this->_dims.end(); ++itr)
        shape_str += std::to_string(*itr) + " ";

    return shape_str + "\n";
}

template<class T>
Tensor<T> Tensor<T>::operator*=(float multiplier)
{
    for (int i = 0; i < this->_size; i++)
        this->_values[i] *= multiplier;

    return *this;
}

template<class T>
Tensor<T> Tensor<T>::operator/=(float divisor)
{
    for (int i = 0; i < this->_size; i++)
        this->_values[i] /= divisor;

    return *this;
}


template<class T>
Tensor<T> Tensor<T>::operator+=(float addend)
{
    for (int i = 0; i < this->_size; i++)
        this->_values[i] += addend;

    return *this;
}

template<class T>
Tensor<T> Tensor<T>::operator*(float multiplier)
{
    auto new_tensor = Tensor<T>(*this);
    new_tensor *= multiplier;
    return new_tensor;
}

template<class T>
Tensor<T> Tensor<T>::operator/(float divisor)
{
    auto new_tensor = Tensor<T>(*this);
    new_tensor /= divisor;
    return new_tensor;
}

template<class T>
Tensor<T> Tensor<T>::operator+(float addend)
{
    auto new_tensor = Tensor<T>(*this);
    new_tensor += addend;
    return new_tensor;
}

template<class T>
template<class U>
Tensor<T> Tensor<T>::operator*(Tensor<U> tensor)
{
    T temp[this->_size];
    auto vals1 = this->_values;
    auto vals2 = tensor._values;

    // Element wise multiplication
    for (int i = 0; i < tensor._size; i++)
        temp[i] = vals1[i] * vals2[i];

    return Tensor<T>(temp, this->_dims, this->_size);
}

template<class T>
Tensor<T> Tensor<T>::transpose()
{
    T temp[this->_size];
    std::vector<int> transpose_dims;

    if (this->_dims.size() == 3)
    {
        int batch_size, rows, cols, matrix_size;

        batch_size = this->_dims[0];
        rows = this->_dims[1];
        cols = this->_dims[2];
        matrix_size = rows * cols;

        // Copy values
        transpose_dims = this->_dims;

        // swap row and col dims
        transpose_dims[1] = cols;
        transpose_dims[2] = rows;


        // Write transposed values into temporary array
        for (int i = 0; i < batch_size; i++)
        {
            // temporary array to hold transpose of one matrix
            T matrix[matrix_size];

            // Offset to find correct array in inner representation
            int offset = matrix_size * i;
            _transpose(rows, cols, this->_values + offset, matrix);

            // Copy matrix into larger temp
            memcpy(temp + offset, matrix, matrix_size * sizeof(T));
        }
    }
    else if (this->_dims.size() == 2)
    {
        int rows, cols, matrix_size;

        rows = this->_dims[0];
        cols = this->_dims[1];

        transpose_dims = {cols, rows};

        _transpose(rows, cols, this->_values, temp);
    }

    return Tensor<T>(temp, transpose_dims, this->_size);
}

template<class T>
Tensor<T>::~Tensor()
{
    delete[] this->_values;
}

template<class T>
Tensor<T> Tensor<T>::operator=(Tensor<T> tensor)
{
    delete this->_values;
    this->_values = new T[tensor._size];
    this->_dims = std::vector<int>();
    this->_dims = tensor._dims;
    this->_size = tensor._size;

    memcpy(this->_values, tensor._values, tensor._size * sizeof(T));
    return *this;
}

template<class T>
template<class U>
Tensor<T> Tensor<T>::operator+(Tensor<U> tensor)
{
    assert(tensor._dims[2] == this->_dims[2]);

    int batch_size, num_cols;
    T temp[this->_size];

    batch_size = this->_dims[0];
    num_cols = this->_dims[2];

    memcpy(temp, this->_values, this->_size * sizeof(T));

    bool same_dims = true;
    for(int i = 0; i < this->_dims.size(); i++)
    {
        if (!(same_dims=(this->_dims[i] == tensor._dims[i])))
            break;
    }

    // Element wise
    if (same_dims)
    {
        for (int i = 0; i < this->_size; i++)
            temp[i] -= tensor._values[i];
    }
    else
    {
        for (int i = 0; i < batch_size; i++) {
            int offset;
            offset = i * num_cols;


            for (int j = 0; j < num_cols; j++) {
                T add = tensor._values[j];
                (temp + offset)[j] += add;
            }
        }
    }

    return Tensor<T>(temp, this->_dims, this->_size);
}

template<class T>
Tensor<T> Tensor<T>::squeeze()
{
    std::vector<int> vec;
    vec = this->_dims;
    vec.erase(std::remove(vec.begin(), vec.end(), 1), vec.end());

    return Tensor<T>(this->_values, vec, this->_size);
}

template<class T>
T Tensor<T>::mean()
{
    auto summed = this->sum();
    return summed / this->_size;
}

template<class T>
T Tensor<T>::sum()
{
    T sum = 0;

    for(int i = 0; i < this->_size; i++)
        sum += this->_values[i];

    return sum;
}

template<class T>
Tensor<T> Tensor<T>::argmax()
{
    // only works for 1 case right now
    T reduced[this->_dims[0]];
    int num_class = this->_dims[1];
    int batch_size = this->_dims[0];

    for(int i = 0; i < batch_size; i++)
    {
        int offset = i * num_class;
        T max = -INFINITY;
        int max_idx = 0;

        for(int j = 0; j < num_class; j++)
        {
            T val = (this->_values + offset)[j];

            if(val > max)
            {
                max_idx = j;
                max = val;
            }
            reduced[i] = max_idx;
        }
    }

    return Tensor<T>(reduced, {batch_size}, batch_size);
}

template<class T>
template<class U>
Tensor<T> Tensor<T>::operator-(Tensor<U> tensor)
{
    assert(tensor._dims[2] == this->_dims[2]);

    int batch_size, num_cols;
    T temp[this->_size];

    batch_size = this->_dims[0];
    num_cols = this->_dims[2];

    memcpy(temp, this->_values, this->_size * sizeof(T));

    bool same_dims = true;
    for(int i = 0; i < this->_dims.size(); i++)
    {
        if (!(same_dims=(this->_dims[i] == tensor._dims[i])))
            break;
    }

    // Element wise
    if (same_dims)
    {
        for (int i = 0; i < this->_size; i++)
            temp[i] -= tensor._values[i];
    }
    else
    {
        for (int i = 0; i < batch_size; i++) {
            int offset;
            offset = i * num_cols;


            for (int j = 0; j < num_cols; j++) {
                T sub = tensor._values[j];
                (temp + offset)[j] -= sub;
            }
        }
    }

    return Tensor<T>(temp, this->_dims, this->_size);
}

template<class T>
Tensor<T> Tensor<T>::ln()
{
    T temp[this->_size];
    memcpy(temp, this->_values, this->_size * sizeof(T));

    for(int i = 0; i < this->_size; i++)
        temp[i] = log(temp[i]);

    return Tensor<T>(temp, this->_dims, this->_size);
}

template<class T>
Tensor<T> Tensor<T>::clip(T low, T high)
{
    T temp[this->_size];
    memcpy(temp, this->_values, this->_size * sizeof(T));

    for(int i = 0; i < this->_size; i++)
    {
        if (temp[i] < low)
            temp[i] = low;
        else if (temp[i] > high)
            temp[i] = high;
    }

    return Tensor<T>(temp, this->_dims, this->_size);
}

template<class T>
Tensor<float> Tensor<T>::operator==(Tensor<T> tensor)
{
    float temp[this->_size] = {0};
    for(int i = 0; i < this->_size; i++)
        temp[i] = float(tensor._values[i] == this->_values[i]);

    return Tensor<T>(temp, this->_dims, this->_size);
}

template<class T>
Tensor<T> Tensor<T>::sum(int dim)
{

    // only works for 1 case right now

    std::vector<int> reduced_dims;
    reduced_dims = this->_dims;
    reduced_dims[dim] = 1;

    int reduced_size = this->_size / this->_dims[dim];
    int num_feat = this->_dims[2];
    T temp[num_feat] = {0};

    for (int i = 0; i < this->_dims[0]; i++)
    {
        int offset = i * num_feat;
        for (int j = 0; j < num_feat; j++)
            temp[j] += (this->_values + offset)[j];
    }

    return Tensor<T>(temp, reduced_dims, reduced_size);
}

template<class T>
Tensor<T> Tensor<T>::unsqueeze(int dim)
{
    std::vector<int> new_dims;
    new_dims = this->_dims;
    new_dims.insert(new_dims.begin() + dim, 1);

    return Tensor<T>(this->_values, new_dims, this->_size);
}

template<class T>
Tensor<T> zeros(const std::vector<int> dims)
{
    int total_size = 1;

    for(int num : dims)
    {
        total_size *= num;
    }

    T val[total_size] = {0};

    return Tensor<T>(val, dims, total_size);
}

