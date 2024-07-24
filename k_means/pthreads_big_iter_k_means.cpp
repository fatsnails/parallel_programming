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
pthread_barrier_t barrier;

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
    int cen_start,cen_end;
    double* newCost,*oldCost;
};

void* threadKMeans(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    double newCost = 0.0;
    int maxIterations = 100;
    do{
        pthread_barrier_wait(&barrier);
        *args->oldCost = *args->newCost;
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
        *args->newCost += newCost;
        pthread_barrier_wait(&barrier);
        for (int i = args->cen_start; i < args->cen_end; i++){
            int count = 0;
            (*args->centroids)[i].r = (*args->centroids)[i].g = (*args->centroids)[i].b = 0;
            for (int j = 0; j < args->pixels->size(); j++) {
                if ((*args->pixels)[j].cluster == i) {
                    (*args->centroids)[i].r += (*args->pixels)[j].r;
                    (*args->centroids)[i].g += (*args->pixels)[j].g;
                    (*args->centroids)[i].b += (*args->pixels)[j].b;
                    count++;
                }
            }
            if (count > 0) {
                (*args->centroids)[i].r /= count;
                (*args->centroids)[i].g /= count;
                (*args->centroids)[i].b /= count;
            }
        }
        pthread_barrier_wait(&barrier);
    }while(fabs(*args->newCost - *args->oldCost) > 1e-6 && --maxIterations > 0);
    return nullptr;
}

void parallelKMeans(vector<Pixel>& pixels, int numClusters, vector<Pixel>& centroids) {
    int numPixels = pixels.size();
    double oldCost, newCost = 0.0;
    int numThreads = 8;
    pthread_t threads[numThreads];
    ThreadArgs args[numThreads];
    pthread_barrier_init(&barrier,NULL,numThreads);
    for (int i = 0; i < numThreads; i++) {
        args[i].pixels = &pixels;
        args[i].centroids = &centroids;
        args[i].start = i * (numPixels / numThreads);
        args[i].end = (i + 1) * (numPixels / numThreads);
        args[i].cen_start = i*(numClusters/numThreads);
        args[i].cen_end = (i+1)*(numClusters/numThreads);
        if (i == numThreads - 1) {
            args[i].end = numPixels;
            args[i].cen_end = numClusters;
        }
        args[i].newCost = &newCost;
        args[i].oldCost = &oldCost;
        pthread_create(&threads[i], nullptr, threadKMeans, &args[i]);
    }
    for (int i = 0; i < numThreads; i++) {
        pthread_join(threads[i], nullptr);
    }
    pthread_barrier_destroy(&barrier);    
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
    // 并行K-Means聚类
    parallelKMeans(pixels, numClusters, centroids);
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