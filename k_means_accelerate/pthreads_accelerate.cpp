#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <pthread.h>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <chrono>

using namespace std;

// 像素点结构体
struct Pixel {
    int r, g, b;
    int cluster;
};

// 计算两个像素点之间的欧氏距离
double distance(Pixel* p1, Pixel* p2) {
    return sqrt(pow(p1->r - p2->r, 2) + pow(p1->g - p2->g, 2) + pow(p1->b - p2->b, 2));
}

// 初始化质心，这里用的方法是在所有的颜色点中随机抽取一个作为质心
void initCentroids(vector<Pixel>& pixels, int numClusters, vector<Pixel>& centroids) {
    int m = pixels.size();
    time_t t;
    srand((unsigned) time(&t));
    int ind;
    for (int i = 0; i < numClusters; i++) {
        ind = rand()%m;
        centroids[i].r = pixels[ind].r;
        centroids[i].g = pixels[ind].g;
        centroids[i].b = pixels[ind].b;
        centroids[i].cluster = i;
    }
}

// Pthreads 版本的 K-Means 聚类算法
struct ThreadArgs {
    vector<Pixel>* pixels;
    vector<Pixel>* centroids;
    int start, end;
};

void* threadKMeans(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    double newCost = 0.0;
    for (int i = args->start; i < args->end; i++) {
        double min = INFINITY;
        for (int j = 0; j < args->centroids->size(); j++) {
            double dist = distance(&(*args->pixels)[i], &(*args->centroids)[j]);
            if (dist < min) {
                min = dist;
                (*args->pixels)[i].cluster = j;
            }
        }
        newCost += min;
    }
    return nullptr;
}

void parallelKMeans(vector<Pixel>& pixels, int numClusters, vector<Pixel>& centroids) {
    int numPixels = pixels.size();
    int maxIterations = 500;
    do {

        // 将像素点分配到最近的质心，使用 Pthreads 并行
        int numThreads = 16;
        pthread_t threads[numThreads];
        ThreadArgs args[numThreads];
        for (int i = 0; i < numThreads; i++) {
            args[i].pixels = &pixels;
            args[i].centroids = &centroids;
            args[i].start = i * (numPixels / numThreads);
            args[i].end = (i + 1) * (numPixels / numThreads);
            if (i == numThreads - 1) {
                args[i].end = numPixels;
            }
            pthread_create(&threads[i], nullptr, threadKMeans, &args[i]);
        }
        for (int i = 0; i < numThreads; i++) {
            pthread_join(threads[i], nullptr);
        }

        // 更新质心位置，这里对质心做并行
        for (int j = 0; j < numClusters; j++) {
            int count = 0;
            centroids[j].r = centroids[j].g = centroids[j].b = 0;
            for (int i = 0; i < numPixels; i++) {
                if (pixels[i].cluster == j) {
                    centroids[j].r += pixels[i].r;
                    centroids[j].g += pixels[i].g;
                    centroids[j].b += pixels[i].b;
                    count++;
                }
            }
            if (count > 0) {
                centroids[j].r /= count;
                centroids[j].g /= count;
                centroids[j].b /= count;
            }
        }
    } while (--maxIterations > 0);
}

int main() {
    // 载入JPG图像并初始化像素点集合
    cv::Mat img = cv::imread("fruit.jpg");
    int width = img.cols, height = img.rows;
    int numPixels = width * height;
    vector<Pixel> pixels(numPixels);

    for (int i = 0; i < numPixels; i++) {
        int x = i % width, y = i / width;
        pixels[i].r = img.at<cv::Vec3b>(y, x)[2];
        pixels[i].g = img.at<cv::Vec3b>(y, x)[1];
        pixels[i].b = img.at<cv::Vec3b>(y, x)[0];
    }

    // 初始化质心
    int numClusters = 8;
    vector<Pixel> centroids(numClusters);
    initCentroids(pixels, numClusters, centroids);
    auto start = std::chrono::high_resolution_clock::now();
    // 并行K-Means聚类
    parallelKMeans(pixels, numClusters, centroids);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    cout<<0.000001*duration<<endl;
    // 根据聚类结果生成分割后的图像
    cv::Mat output(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < numPixels; i++) {
        int x = i % width, y = i / width;
        output.at<cv::Vec3b>(y, x)[2] = centroids[pixels[i].cluster].r;
        output.at<cv::Vec3b>(y, x)[1] = centroids[pixels[i].cluster].g;
        output.at<cv::Vec3b>(y, x)[0] = centroids[pixels[i].cluster].b;
    }

    // 保存分割后的图像
    cv::imwrite("output.jpg", output);

    return 0;
}