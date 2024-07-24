#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "mpi.h"
#include <opencv2/opencv.hpp>
#include <ctime>
#include <chrono>

using namespace std;

// 像素点结构体
struct Pixel {
    int r, g, b;
    int cluster;
};

// 计算像素点和中心点之间的距离的，用来给数据点进行分组用的
double distance(Pixel* p1, Pixel* p2) {
    return sqrt(pow(p1->r - p2->r, 2) + pow(p1->g - p2->g, 2) + pow(p1->b - p2->b, 2));
}

// 初始化质心，这里用的方法是在所有的颜色中随机抽取一个作为质心
void initCentroids(vector<Pixel>& pixels, int numClusters, vector<Pixel>& centroids) {
    int m = pixels.size();
    time_t t;
    srand((unsigned) time(&t));
    int ind;
    int i;
    for (i = 0; i < numClusters; i++) {
        ind = rand()%m;
        centroids[i].r = pixels[ind].r;
        centroids[i].g = pixels[ind].g;
        centroids[i].b = pixels[ind].b;
        centroids[i].cluster = i;
        cout<<"("<<centroids[i].r<<","<<centroids[i].g<<","<<centroids[i].b<<")"<<endl;
    }
}

// 并行K-Means算法
void parallelKMeans(vector<Pixel>& pixels, int numClusters, vector<Pixel>& centroids) {
    int numPixels = pixels.size();
    int maxIterations = 100;
    double oldCost, newCost = 0.0;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // 这里do-while是为了在第一次oldcost=newcost的时候不用另外再写代码
    do {
        oldCost = newCost;
        newCost = 0.0;
        double min;
        // 将像素点分配到最近的质心，这个部分是对每个像素点做并行的，是好的
        for (int i = rank; i < numPixels; i += size) {
            min = INFINITY;
            for (int j = 0; j < numClusters; j++) {
                double dist = distance(&pixels[i], &centroids[j]);
                if (dist < min) {
                    min = dist;
                    pixels[i].cluster = j;
                }
            }
            newCost += min;
        }
        // 在本地根据本地的pixel的暂时归类结果，进行本地质心数据的计算
        vector<Pixel> localCentroids(numClusters);
        vector<int> clusterCounts(numClusters, 0);
        for (int i = rank; i < numPixels; i += size) {
            localCentroids[pixels[i].cluster].r += pixels[i].r;
            localCentroids[pixels[i].cluster].g += pixels[i].g;
            localCentroids[pixels[i].cluster].b += pixels[i].b;
            clusterCounts[pixels[i].cluster]++;
        }
        // 把本地的关于每个质心数据的rgb和团中的像素点数进行汇总，这里是汇总到pid为0的进程，之后pid等于0的进程收到所有关于这个质心的数据就直接进行计算
        for (int i = 0; i < numClusters; i++) {
            int globalCount;
            int red,green,blue;
            MPI_Reduce(&clusterCounts[i], &globalCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&localCentroids[i].r,&red,1,MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&localCentroids[i].g,&green,1,MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&localCentroids[i].b,&blue,1,MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                if (globalCount > 0) {
                    centroids[i].r = red / globalCount;
                    centroids[i].g = green/ globalCount;
                    centroids[i].b = blue / globalCount;
                }
            }
        }
        // 在所有质心的数据都算完之后把新的质心的信息播送给其他的进程，方便后续的计算
        MPI_Bcast(centroids.data(), 3 * numClusters, MPI_INT, 0, MPI_COMM_WORLD);
        // 这里是比较容易忘记，就是在整个算cost的时候，不同的进程算出的可能不一样，在一些进程看起来差不多行了的时候，其他的进程可能还没结束，因此要对这个cost进行汇总之后播送
        double globalCost;
        MPI_Reduce(&newCost, &globalCost, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            newCost = globalCost;
        }
        //MPI_Barrier(MPI_COMM_WORLD);
        //这个是防止有的线程的newcost先和oldcost差不多了，先自己跳出循环了，那进程0就会无限制等待
        MPI_Bcast(&newCost, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } while (fabs(newCost - oldCost) > 1e-6 && --maxIterations > 0);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // 载入JPG图像并初始化像素点集合
    cv::Mat img = cv::imread("fruit.jpg");
    int width = img.cols, height = img.rows;
    int numPixels = width * height;
    vector<Pixel> pixels(numPixels);
    //这里如果进行并行的话会比较麻烦，就直接每个进程自己读了一遍，虽然有重复的计算，但也减少了通信的开销
    for (int i = 0; i < numPixels; i++) {
        int x = i % width, y = i / width;
        pixels[i].r = img.at<cv::Vec3b>(y, x)[2];
        pixels[i].g = img.at<cv::Vec3b>(y, x)[1];
        pixels[i].b = img.at<cv::Vec3b>(y, x)[0];
        //cout<<"("<<pixels[i].r<<","<<pixels[i].g<<","<<pixels[i].b<<")"<<endl;
    }

    // 初始化质心
    int numClusters = 8;
    vector<Pixel> centroids(numClusters);
    if (rank == 0) {
        initCentroids(pixels, numClusters, centroids);
    }
    // 把初始之后的质心播送到所有的进程中
    MPI_Bcast(centroids.data(), 3 * numClusters, MPI_INT, 0, MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    // 并行的算法的进行
    parallelKMeans(pixels, numClusters, centroids);

    double min;
    if (rank == 0) {
        //这里有个易错点，就是每个pixel的聚类一直是分开算的，但是最后应该合起来，这里要用rank=0的进程算一下，虽然把分开算的数据汇总到一起也不是不行，但是会有传输开销
        for (int i = 0; i < numPixels; i ++) {
            min = INFINITY;
            for (int j = 0; j < numClusters; j++) {
                double dist = distance(&pixels[i], &centroids[j]);
                if (dist < min) {
                    min = dist;
                    pixels[i].cluster = j;
                }
            }
        }
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
        // 输出处理后的图像
        cv::imwrite("output.jpg", output);
    }

    MPI_Finalize();
    return 0;
}