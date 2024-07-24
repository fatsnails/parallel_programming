
#include<omp.h>
#include<iostream>
#include<ctime>
#include<random>


using namespace std;


//这个step是并行代码和串行代码共用的
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

//并行版本的quicksort
void quicksort(int* temp,int left,int right){
    if(left>=right) return;
    int k = step(temp,left,right);
    #pragma omp task
    {
    quicksort(temp,left,k);
    }
    #pragma omp task
    {
    quicksort(temp,k+1,right);
    }
}

//串行版本的quicksort
void quicksort_serial(int* temp,int left,int right){
    if(left>=right) return;
    int k = step(temp,left,right);
    quicksort_serial(temp,left,k);
    quicksort_serial(temp,k+1,right);
}

int main(){
    int N = 10000000;//数据的量
    int iter = 30;//重复实验的次数
    int i,j;
    double total_duration = 0.0;
    double total_duration_serial = 0.0;
    int*p = new int[N];
    int*q = new int[N];
    time_t t;
    srand((unsigned) time(&t));
    int temp;//用于生成数据的

    for (i = 0;i < iter;i++){
        for(j = 0;j < N;j++){
            temp = rand();
            p[j] = temp;
            q[j] = temp;
        }

        cout<<endl<<"iteration:"<<i+1<<endl;

        omp_set_num_threads(4);//这个是设置运行的线程数的
        double start = omp_get_wtime();
        #pragma omp parallel
        {
        #pragma omp single nowait
        quicksort(p,0,N);
        }
        double end = omp_get_wtime();
        cout<<"duration:"<<end-start<<"s"<<endl;
        total_duration+=end-start;
        /*
        for(j = 0;j < N;j++){
            cout<<p[j]<<" ";
        }
        cout<<endl;*/

        double start_serial = omp_get_wtime();
        quicksort_serial(q,0,N);
        double end_serial = omp_get_wtime();
        cout<<"serial duration:"<<end_serial-start_serial<<"s"<<endl;
        total_duration_serial+=end_serial-start_serial;
    }

    cout<<"avg_duration:"<<double(total_duration)/iter<<"s"<<endl;
    cout<<"avg_duration_serial:"<<double(total_duration_serial)/iter<<"s"<<endl;
    cout<<"the speedup is calculated as:"<< double(total_duration_serial)/double(total_duration)<<endl;
    return 0;
}