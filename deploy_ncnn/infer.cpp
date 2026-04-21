#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "net.h"

namespace
{
constexpr int kInputSize = 64;
constexpr int kTopK = 3;

std::vector<std::string> load_labels(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open labels file: " + path);
    }

    std::vector<std::string> labels;
    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty())
        {
            labels.push_back(line);
        }
    }
    return labels;
}

std::vector<float> softmax(const std::vector<float>& logits)
{
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> exps(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i)
    {
        exps[i] = std::exp(logits[i] - max_logit);
        sum += exps[i];
    }
    for (float& value : exps)
    {
        value /= sum;
    }
    return exps;
}
}

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0]
                  << " model.param model.bin image.jpg labels.txt" << std::endl;
        return 1;
    }

    const std::string param_path = argv[1];
    const std::string bin_path = argv[2];
    const std::string image_path = argv[3];
    const std::string labels_path = argv[4];

    const std::vector<std::string> labels = load_labels(labels_path);

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return 1;
    }

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;

    if (net.load_param(param_path.c_str()) != 0)
    {
        std::cerr << "Failed to load param: " << param_path << std::endl;
        return 1;
    }
    if (net.load_model(bin_path.c_str()) != 0)
    {
        std::cerr << "Failed to load bin: " << bin_path << std::endl;
        return 1;
    }

    // BGR mean/std aligned with the training pipeline.
    const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[3] = {
        1.0f / 57.375f,
        1.0f / 57.12f,
        1.0f / 58.395f,
    };

    const auto t0 = std::chrono::high_resolution_clock::now();

    ncnn::Mat input = ncnn::Mat::from_pixels_resize(
        image.data,
        ncnn::Mat::PIXEL_BGR,
        image.cols,
        image.rows,
        kInputSize,
        kInputSize);
    input.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("input", input);

    ncnn::Mat output;
    if (ex.extract("logits", output) != 0)
    {
        std::cerr << "Failed to extract output tensor." << std::endl;
        return 1;
    }

    std::vector<float> logits(output.w);
    for (int i = 0; i < output.w; ++i)
    {
        logits[i] = output[i];
    }

    const std::vector<float> probs = softmax(logits);

    std::vector<int> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int lhs, int rhs) {
        return probs[lhs] > probs[rhs];
    });

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

    std::cout << "Image: " << image_path << std::endl;
    std::cout << "End-to-end time: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms" << std::endl;

    const int topk = std::min(static_cast<int>(indices.size()), kTopK);
    for (int rank = 0; rank < topk; ++rank)
    {
        const int cls = indices[rank];
        const std::string label = cls < static_cast<int>(labels.size()) ? labels[cls] : "unknown";
        std::cout << "Top" << (rank + 1) << ": " << label << " (" << probs[cls] << ")" << std::endl;
    }

    return 0;
}
