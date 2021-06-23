#include <iostream>
#include "Network.h"
#include "Tensor.h"

#include <random>

int main()
{
//    int test[3][4][1] = {0};
//    Tensor<int> test_tensor = Tensor<int>();
//    auto flatten = [](auto arr) {return reinterpret_cast<int(*)[12]>(arr);};
//    auto newt = flatten(test);
//
//    for (int i = 0; i < 12; i++)
//    {
//        std::cout << *(newt)[0];
//    }
//


//    float test[18] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
//    std::vector<int> dims = {2, 3, 3};
//    auto tensor = new Tensor<float>(test, dims);
//    auto tensor_o = *tensor;
//
//    // std::cout << tensor;
//
//    std::cout << tensor->shape() << "\n";
//    std::cout << *tensor;
//
//    tensor_o = tensor_o * 5;
//
//    std::cout << tensor_o;
//
//    tensor_o = tensor_o + 5;
//
//    std::cout << tensor_o;


//    int A[6] = {0, 1, 2, 3, 4, 5};
//    int B[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11};
//    int C[8] = {0};
//
//    mm(A, B, C, 2, 3, 4);
//
//    for (int i = 0; i < 8; i++)
//    {
//        std::cout << C[i] << " ";
//    }

//    int test[10] = {0};
//    range(21, 31, test);
//    for (int i = 0; i < 10; i++)
//    {
//        std::cout << test[i] << " ";
//    }

//    float test[18] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
//    std::vector<int> dims = {2, 3, 3};
//
//
//
//    // std::cout << tensor;
//
//    std::cout << tensor->shape() << "\n";
//    std::cout << *tensor;
//
//    tensor_o = tensor_o * 5;
//
//    std::cout << tensor_o;
//
//    tensor_o = tensor_o + 5;
//
//    std::cout << tensor_o;

//    float A[12], B[12];
//    range(0, 12, A);
//    range(0, 12, B);
//
//    auto tensor_A = Tensor<float>(A, {2, 2, 3}, 12);
//    auto tensor_B = Tensor<float>(B, {1, 3, 4}, 12);
//
//    std::cout << tensor_A << std::endl;
//    std::cout << tensor_B << std::endl;
//
//    auto tensor_C = bmm(tensor_A, tensor_B);
//
//    std::cout << tensor_C << std::endl;
//
//    auto tensor_D = tensor_C.reshape({2, 4, 2});
//
//    std::cout << tensor_D << std::endl;
//
//    auto tensor_E = tensor_C.transpose();
//
//    std::cout << tensor_E << std::endl;

//    int Q[6] = {0, 1, 2, 3, 4, 5};
//    int Q2[6] = {0};
//    transpose(2, 3, Q, Q2);
//
//    for (int i = 0; i < 6; i++)
//    {
//        std::cout << Q2[i] << " ";
//    }


//    std::normal_distribution<> d(0, 1);
//    std::random_device rd{};
//    std::mt19937 gen{rd()};
//
//    for(int n=0; n<100; ++n) {
//        std::cout << d(gen) << " ";
//    }

//    Linear lin = Linear(3, 5);
//    std::cout << *(lin._weights) << "oook";


//    float A[18], B[12], C[32], D[24];
//    range(0, 18, A);
//    range(0, 12, B);
//    range(0, 32, C);
//    range(0, 24, D);
//
//    auto tA = Tensor<float>(A, {6, 1, 3}, 18);
//    auto tB = Tensor<float>(B, {1, 3, 4}, 12);
//    auto tC = Tensor<float>(C, {1, 4, 8}, 32);
//    auto tD = Tensor<float>(D, {1, 8, 3}, 24);
//
//    auto E = bmm(tA, tB);
//    E = bmm(E, tC);
//    E = bmm(E, tD);
//
//    std::cout << E.shape();
//    std::cout << E;

    float A[18];
    range(0, 18, A);
    auto tA = Tensor<float>(A, {6, 1, 3}, 18);

    Network ffn;
    auto out = ffn(tA);
    std::cout << out;

}
