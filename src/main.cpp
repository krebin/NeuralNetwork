#include <iostream>
#include "Network.h"
#include "Tensor.h"
#include "Loss.h"
#include "utils.h"

#include <random>
#include <vector>
#include <algorithm>

int main()
{
    int num_samp = 60000;
    int num_feat = 28 * 28;
    int num_class = 10;
    std::string train_label_path = "data/train-labels-idx1-ubyte";
    std::string train_data_path = "data/train-images-idx3-ubyte";

    // Load MNIST
    // https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
    unsigned char *labels_uchar = read_mnist_labels(train_label_path, num_samp);
    unsigned char **images_uchar = read_mnist_images(train_data_path, num_samp, num_feat);

    float Y_original[num_samp];
    // One hot encoding
    float *Y[num_samp];
    float *X[num_samp];

    // Convert to float arrays
    for (int i = 0; i < num_samp; i++)
    {
        // Allocate array for ith sample
        X[i] = new float[num_feat];
        Y[i] = new float[num_class]{};

        auto X_i = X[i];
        auto Y_i = Y[i];
        auto images_i = images_uchar[i];

        // Divide by 255 for pixel intensity and convert to float
        for (int j = 0; j < num_feat; j++)
            X_i[j] = float(images_i[j]) / 255;

        delete[] images_i;

        int cls = int(labels_uchar[i]);

        // Original encoding
        Y_original[i] = cls;

        // One hot encoding
        Y_i[cls] = 1;
    }
    delete[] images_uchar;

    Network ffn;
    Loss crossent;
    int batch_size, num_epochs, num_batches;
    float learning_rate;

    batch_size = 200;
    num_batches = num_samp / batch_size;
    num_epochs = 20;

    learning_rate = 0.005;

    std::vector<int> indices(num_samp);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());

    for (int epoch = 0; epoch < num_epochs; epoch++)
    {
        if (epoch > 13)
            learning_rate *= 0.1;

        float loss, acc;

        acc = 0;
        loss = 0;

        // Shuffle indices for training
        std::shuffle(indices.begin(), indices.end(), rd);
        for (int batch = 0; batch < num_batches; batch++)
        {
            auto batch_start = indices.cbegin() + batch_size * (batch);
            auto batch_end = indices.cbegin() + batch_size * (batch + 1);
            auto batch_inds = std::vector<int>(batch_start, batch_end);

            // temporary arrays to turn into tensors
            float temp_X[batch_size * num_feat];
            float temp_Y[batch_size * num_class];
            float temp_Y_original[batch_size];
            float batch_loss = 0;

            // Copy data into temp arrays for tensors
            for(int i = 0; i < batch_size; i++)
            {
                int offset_X, offset_Y, idx;

                offset_X = i * num_feat;
                offset_Y = i * num_class;

                idx = batch_inds[i];

                memcpy(temp_X + offset_X, X[idx], sizeof(float) * num_feat);
                memcpy(temp_Y + offset_Y, Y[idx], sizeof(float) * num_class);

                temp_Y_original[i] = Y_original[idx];
            }

            // Make batch tensors
            auto batch_X = Tensor<float>(temp_X, {batch_size, 1, num_feat}, batch_size * num_feat);
            auto batch_Y = Tensor<float>(temp_Y, {batch_size, 1, num_class}, batch_size * num_class);
            auto batch_Y_o = Tensor<float>(temp_Y_original, {batch_size}, batch_size);

            // ln(x) = -inf , x->0
            auto out = ffn(batch_X).clip(0.0001, 1.0);

            batch_loss = crossent(batch_Y, out);
            loss += batch_loss;

            ffn.backward(out - batch_Y);
            ffn.optimize(learning_rate);

            out = out.squeeze();
            out = out.argmax();

            float batch_acc = (out == batch_Y_o).sum();
            acc += batch_acc;
            batch_acc /= batch_size;

            std::cout
            << "Epoch: " << std::to_string(epoch)
            << " Batch: " << batch
            << " Acc: " << batch_acc
            << " Loss: " << batch_loss
            << std::endl;
        }

        acc /= num_samp;
        loss /= num_batches;

        std::cout
        << std::endl
        << "Epoch " << std::to_string(epoch)
        << " Acc: " << acc
        << " Loss: " << loss
        << std::endl;
    }
}
