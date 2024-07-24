#include <iostream>
#include <vector>
#include <ctime>
#include <algorithm>
#include <chrono>

#include "mpi.h"

using namespace std;

//这个函数是用来给快速排序的一步进行实现的，也就是partition的功能，这两个函数用的是第一题的，就是是串行的版本
int step(int* temp,int left,int right){
    int target = temp[right-1];
    int j = left; //这个是小于主元和大于等于主元的分界
    //这个是小于主元的部分，应该在主元的左侧
    while(j<right&&temp[j]<=target){
        j++;
    }
    int i = j;
    while(i<right){
        if(temp[i]<=target){
            swap(temp[i],temp[j]);
            j++;
        }
        i++;
    }
    return j-1;
}

//这个函数是快速排序，在局部排序和样本排序的时候会用到
void sort_num(int* temp,int left,int right){
    if(left>=right) return;
    int k = step(temp,left,right);
    sort_num(temp,left,k);
    sort_num(temp,k+1,right);
}

//在分块之后的局部排序和样本选取
void local_sort_extract_sample(int*num,int* sample,int s,int e,int p){//这里的这个s是start_ind, e是end_ind ,p是进程数,sample是发出去的样本
    //这里怕不让用库函数，就直接调用的快速排序的函数
    int i;
    sort_num(num,s,e);
    int gap = (e-s)/p; //向下取整
    for(i = 0;i < p;i++){
        sample[i] = num[s+gap*i];
    }
    return;
}

//之前会把sample合并起来，得到级联的gather_sample，这个函数只有pid=0的进程会调用
void sort_sample_extract_pivot(int* gather_sample,int* partition,int p){
    sort_num(gather_sample,0,p*p);
    int i;
    for(i = 1;i < p;i++){
        partition[i - 1] = gather_sample[i*p];
    }
    return;
}

//得到partition之后每个进程要把对应段的数据送到对应的进程，并接到对应的进程并进行排序
//num是排序之前的数组，在这个过程中是只读的，an是要返回的数组，这个数组的存在是因为很难在原数组的基础之上进行排序
//begin是在这个进程中，每一个段的对应的起始位置，part_length是每一段的长度，p是进程的个数
//total_length是这一段的总的要进行merge的长度
void local_final_sort(int*num,int *an,int p,int total_length,int *begin,int *part_length){
    int min,min_ind;
    int i;
    int total_count = 0;//这个是已经merge的总长度
    int *count = new int[p];//每一段已经merge的长度
    for(i = 0;i < p;i++){//这个玩意不会自动初始化，很坑人
        count[i] = 0;
    }
    while(total_count<total_length){
        min = INT_MAX;
        min_ind = -1;
        for(i = 0;i < p;i++){
            if(count[i]<part_length[i]&&num[begin[i]+count[i]]<min){
                min = num[begin[i]+count[i]];
                min_ind = i;
            }
        }
        an[total_count] = min;
        total_count++;
        count[min_ind]++;
    }
    //cout<<endl;
    delete []count;
    return;
}

//把和这个算法有关系的几乎所有的mpi的操作都放在这个函数里，这个函数可能会直接或者间接调用上边定义的函数
double sort(int*num,int N){
    //基本的信息
    int p;
    MPI_Comm_size(MPI_COMM_WORLD,&p);
    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD,&pid);

    //一些准备工作
    int gap =  N/p;
    int s = gap*pid;//id号从0开始，这个是每个进程的开始位置
    int e = gap*(pid+1);//这个是每个进程的结束位置，因为最后一个进程可能会要把剩余的数也带着，因此，对最后一个进行要进行修改
    if(pid == p-1){
        e = N;
    }
    int i,j;
    //if(pid==0){这个是排序之前的输出，由于这个在数比较多的时候会输出很慢，很长，就注释掉了，否则可以把注释去掉，看到确实是可以进行正常排序的
        //for(i = 0;i < N;i++){
        //    cout<< num[i]<<" ";
        //}
        //cout<<endl<<"end of original number"<<endl;
    //}
    int total_length = 0;//这个是每个进程二次排序的时候的元素个数
    int* sample = new int[p];
    int* concat_sample = new int[p*p];
    int* partition = new int[p-1];
    int* send_length = new int[p];//局部排序之后的每一段应该发送的长度，这些都是原有的段内的位置
    int* send_begin = new int[p];//这个是发送的段的起始位置，这些都是原有的段内的位置
    int* begin = new int[p];//每一段的起始位置
    int* part_length = new int[p];
    int* arr_total_length = new int[p];//这个是每个进程的二次排序的元素个数的集合
    int* insert_p = new int[p];//这个是针对于全体数据的这个每一段元素的插入的起始位置
    double end = 0;
    //计时开始
    double start = MPI_Wtime();
    //这步是局部的一个排序并抽取样本的过程
    local_sort_extract_sample(num,sample,s,e,p);

    //如果是单个进程的串行的逻辑，则直接可以返回了，已经排好了顺序
    if(p<2){
        end = MPI_Wtime();
        delete [] sample;
        delete [] concat_sample;
        delete [] partition;
        delete [] send_length;
        delete [] send_begin;
        delete [] begin;
        delete [] part_length;
        delete [] arr_total_length;
        delete [] insert_p;
        cout<<"parallel time: "<< end-start<<endl;
        return end-start;
    }

    //这步是把样本发送到pid=0的进程中进行排序并抽取主元的过程
    MPI_Gather(sample,p,MPI_INT,concat_sample,p,MPI_INT,0,MPI_COMM_WORLD);
    if(pid==0){
        sort_sample_extract_pivot(concat_sample,partition,p);
    }

    //这步是把主元partition进行分发的过程
    MPI_Bcast(partition,p-1,MPI_INT,0,MPI_COMM_WORLD);
    
    send_begin[0] = s;//每个进程根据partition来把自己的数据段进行切分，从开始下标s开始
    send_length[0] = upper_bound(num+s, num+e, partition[0]) - (num+s);
    for(i = 1;i < p;i++){
        send_begin[i] = send_begin[i-1] + send_length[i-1]; 
        if(i >= p - 1){
            break;
        }
        send_length[i] = upper_bound(num+s, num+e, partition[i]) - (num+send_begin[i]);
    }
    send_length[p-1] = e - send_begin[p-1];


    //这一步是把这些数据段的长度播送给对应的进程，并从各个线程中收到自己要进行处理的数据段的长度，这里的part_length是一个进程得到的每一段的长度，要算出给到一个进程的总长度，得到对应的长度了才能真正的收集数据
    MPI_Alltoall(send_length, 1, MPI_INT, part_length, 1, MPI_INT, MPI_COMM_WORLD); 

    total_length += part_length[0];
    begin[0] = 0;//这个是对收到的数组的开始位置
    for(i = 1;i < p; i++){
        begin[i] = begin[i-1] + part_length[i-1];
        total_length += part_length[i];
    }
    //这一步就是发送具体的数据段，并得到了收到的段的具体的值，存在concat_num中，这里会初始化两个新的数组，因为这个total_length不能够提前算出，不能和前面的写到一起
    int* concat_num = new int[total_length];
    int* an = new int[total_length];
    MPI_Alltoallv(num, send_length, send_begin, MPI_INT, concat_num, part_length, begin, MPI_INT, MPI_COMM_WORLD); 
    local_final_sort(concat_num,an,p,total_length,begin,part_length);
    MPI_Gather(&total_length, 1, MPI_INT, arr_total_length,1, MPI_INT,0,MPI_COMM_WORLD);
    if(pid == 0){
        insert_p[0] = 0;
        for(i = 1;i < p;i++){
            insert_p[i] = insert_p[i-1] + arr_total_length[i-1];
        }
    }

    MPI_Gatherv(an,total_length,MPI_INT,num,arr_total_length,insert_p,MPI_INT,0,MPI_COMM_WORLD);
    
    end = MPI_Wtime();

    //这里就可以进行收尾工作了

    delete [] sample;
    delete [] concat_sample;
    delete [] partition;
    delete [] send_length;
    delete [] send_begin;
    delete [] begin;
    delete [] part_length;
    delete [] arr_total_length;
    delete [] insert_p;

    delete [] concat_num;
    delete [] an;
    
    if(pid==0){
        //cout<<"the sorted:"<<endl;
        //for(i = 0;i < N;i++){
        //    cout<<num[i]<<" ";
        //}
        cout<<"parallel time: "<< end-start<<endl;
    }
    
    return end-start;
}

int main(int argc,char *argv[]){
    int N = 10000000;
    int iter = 30;
    int i,j;
    time_t t;
    srand((unsigned)time(&t));
    int*p = new int[N];
    int*q = new int[N];
    int temp;//用于生成数据的
    double time_exec;
    double time_avg = 0.0;
    double time_avg_serial = 0.0;
    int pid;
    double begin_time;
    double end_time;
    double duration;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&pid);
    for(i = 0;i < iter;i++){
        for(j = 0;j < N;j++){
            temp = rand();
            p[j] = temp;
            q[j] = temp;
        }
        if(pid == 0){
            cout<<endl<< "iteration "<<i<<endl;
        }
        time_exec =sort(p,N);
        if(pid == 0){
            begin_time = MPI_Wtime();
            sort_num(q,0,N);
            end_time = MPI_Wtime();
            duration = end_time - begin_time;
            cout<< "serial time: "<< duration << endl;
        }
        time_avg += time_exec;
        time_avg_serial += duration;
    }
    time_avg /= iter;
    time_avg_serial /= iter;
    if(pid==0){
        cout<<"average execution time: "<<double(time_avg)<<endl;
        cout << "average serial execution time: "<<double(time_avg_serial)<<endl;
        cout<<"the speedup is calculated as: "<<double(time_avg_serial)/double(time_avg)<<endl;
    }
    MPI_Finalize();
    delete []p;
    return 0;
}