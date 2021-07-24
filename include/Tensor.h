//
// Created by krebin on 6/15/21.
//
#include <iostream>
#include <vector>
#include <ostream>

#ifndef NOORALNETWORK_TENSOR_H
#define NOORALNETWORK_TENSOR_H

template <class T>
class Tensor
{
    public:

       /**
        * Constructor
        *
        * @param *vals T array that is copied into a new T array on heap
        * @param dims Vector containing the Tensor dimensions needed to access *vals correctly
        * @param size Optional parameter, size of vals, will skip redundant computation if passed in
        */
        Tensor(T *vals, const std::vector<int>& dims, int size=-1);

        /**
        * Copy constructor
        * Copies _size and _dims, allocates new array on heap for _vals
        *
        * @param tensor Tensor to copy values from
        */
        Tensor(Tensor<T> const &tensor);

        /**
         * Destructor
         */
        ~Tensor();

        std::string shape();

        /**
         * Remove dimensions equal to 1
         *
         * @return Tensor object with 1's removed from dims
         */
        Tensor<T> squeeze();

        /**
         * Add a dimension
         *
         * @param dim axis to add
         *
         * @return Tensor object with additional dimension
         */
        Tensor<T> unsqueeze(int dim);

        /**
         * Element wise natural log
         *
         * @return Tensor object with element wise log applied
         */
        Tensor<T> ln();

        /**
         * Clips elements wise
         *
         * @param low Elements < low set to low
         * @param high Elements > high set to high
         *
         * @return Tensor object with clipped values
         */
        Tensor<T> clip(T low, T high);

        /**
         * Mean of all elements in tensor
         *
         * @return mean of all elements in tensor
         */
        T mean();

        /**
         * Sum of all elements in tensor
         *
         * @return Sum of all elements in tensor
         */
        T sum();

        /**
         * Max index channel wise of tensor
         *
         * @return Max index channel wise of tensor
         */
        Tensor<T> argmax();

        /**
         * Sum of all elements in one dimension of tensor
         *
         * @return Sum of all elements in one dimension of tensor
         */
        Tensor<T> sum(int dim);

        /**
         * Sets tensor equal to input tensor
         *
         * @param tensor Tensor to be copied
         *
         * @return Tensor object with updated values
         */
        Tensor<T>& operator= (Tensor<T> tensor);

        /**
         * Checks for equality element wise of two tensors
         *
         * @param tensor Tensor to check with
         *
         * @return Tensor<bool> object that signifies equality at each element
         */
        Tensor<float> operator== (Tensor<T> tensor);

        /**
         * Multiplies every element in _vals input value. in place version of *
         *
         * @param multiplier value to multiply elements by
         *
         * @return Original Tensor object with modified values
         */
        Tensor<T> operator*= (float multiplier);

        /**
         * Divides every element in _values input value. in place version of /
         *
         * @param divisor value to divide elements by
         *
         * @return Original Tensor object with modified values
         */
        Tensor<T> operator/= (float divisor);

        /**
         * Adds every element in _vals input value. in place version of +
         *
         * @param addend value to add elements by
         *
         * @return Original Tensor object with modified values
         */
        Tensor<T> operator+= (float addend);

        /**
         * Multiplies every element in _vals input value
         *
         * @param multiplier value to multiply elements by
         *
         * @return a new Tensor object with modified values
         */
        Tensor<T> operator* (float multiplier);

        /**
         * Multiplies every element in _vals input value
         *
         * @param divisor value to divide elements by
         *
         * @return a new Tensor object with modified values
         */
        Tensor<T> operator/ (float divisor);

        /**
         * Adds every element in _vals by input value
         *
         * @param addend value to add elements by
         *
         * @return a new Tensor object with modified values
         */
        Tensor<T> operator+ (float addend);


        /**
         * Multiplies element wise every element in this Tensor with input tensor
         * Both Tensors must have same dimensions
         *
         * @param tensor Tensor to element wise multiply with
         *
         * @return a new Tensor object with modified values
         */
        template <class U>
        Tensor<T> operator* (Tensor<U> tensor);

        /**
         * Adds element i in input tensor to every element in row i of this tensor
         *
         * @param tensor Tensor to get values from
         *
         * @return a new Tensor object with modified values
         */
        template <class U>
        Tensor<T> operator+ (Tensor<U> tensor);

        /**
         * Adds element i in input tensor to every element in row i of this tensor
         *
         * @param tensor Tensor to get values from
         *
         * @return a new Tensor object with modified values
         */
        template <class U>
        Tensor<T> operator- (Tensor<U> tensor);


    /**
         * Create new Tensor with new dimensions
         * New size must be same as current size
         *
         * @param dims new vector of dimensions for tensor
         *
         * @return new Tensor with updated dims
         */
        Tensor<T> reshape(std::vector<int> dims);

        /**
         * Returns transpose of matrices in second and third dimensions
         *
         * @return new Tensor with transposed values
         */
        Tensor<T> transpose();

        template <class U>
        friend std::ostream& operator<< (std::ostream &out, const Tensor<U>& tensor);

        /**
        * Batch Matrix multiplication
        * Will matrix multiply every matrix in tensor A with single matrix in tensor B
        *
        * @param A Tensor of size [n, rows_A, cols_A]
        * @param B Tensor of size [1, rows_B, cols_B]
        *
        * @return Tensor of size [n, rows_A, cols_B]
        *
        */
        template <class U>
        friend Tensor<U> bmm(const Tensor<U> &tensor_A, const Tensor<U> &tensor_B);

        // "cpu" by default. will add "cuda" later
        std::string device;

        // Num elements in tensor
        int _size;

        // Inner 1-D representation of tensor
        // all functions assume tensor will be of form [batch_size, rows, cols]
        T *_values;

        // True dimensions of the tensor
        std::vector<int> _dims;

        bool del_vals;

    private:
        friend class Layer;



};

// Tensor zeros(std::vector<int> dims);

#include "../src/Tensor.tpp"
#endif //NOORALNETWORK_TENSOR_H