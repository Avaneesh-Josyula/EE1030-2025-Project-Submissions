#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../c_libs/stb_image.h"
#include "../c_libs/stb_image_write.h"


double* load(const char* filename,int*height,int* width) {
    int channels;
    unsigned char* img = stbi_load(filename,width,height,&channels,1); // channels=1 represents greyscale and 3 represents RGB and 4 represents RGBA
    if(!img){
        fprintf(stderr,"Error");
        return NULL;
    }

    int size =(*width)*(*height);
    double* data =(double*)malloc(size*sizeof(double));
    if(!data){
        fprintf(stderr, "Error in mem aloc for data\n");
        stbi_image_free(img);
        return NULL;
    }

    for(int i=0;i<size;i++)
        data[i]=img[i]/ 255.0;

    stbi_image_free(img);
    return data;
}

void write_png(const char* filename,const double* data,int width,int height) {
    unsigned char* img= (unsigned char*)malloc(width*height);
    if(!img){
        fprintf(stderr,"Error in mem aloc\n");
        return;
    }

    for (int i=0;i< width*height;i++) {
        double val=data[i];
        if(val<0.0) val=0.0;
        if(val>1.0) val=1.0;
        img[i] = (unsigned char)(val*255.0);
    }

    if(stbi_write_png(filename,width,height,1,img,width)== 0)
        fprintf(stderr,"Error in writing file\n");

    free(img);
}

void write_jpg(const char* filename,const double* data,int width,int height) {
    unsigned char* img =(unsigned char*)malloc(width*height);
    if(!img){
        fprintf(stderr,"Error in mem aloc\n");
        return;
    }

    for(int i=0;i<width*height;i++) {
        double val=data[i];
        if (val<0.0) val=0.0;
        if (val>1.0) val=1.0;
        img[i] =(unsigned char)(val*255.0);
    }

    if(stbi_write_jpg(filename,width,height,1,img,90)==0)
        fprintf(stderr,"Error in writing file\n");

    free(img);
}


void copy(double* B,const double* A,int size) {
    for(int i=0;i<size;i++) B[i]=A[i];
}

void transpose(const double* A,double* At,int rows,int cols) {
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            At[j*rows+i] =A[i*cols+j];
}

void mul_mat(const double* A,const double* B,double* C,int m,int p,int n) {
    for(int i=0;i<m;i++)
        for(int j=0;j<n;j++) {
            double s = 0.0;
            for(int k=0;k<p;k++)
                s += A[i*p+k]*B[k*n+j];
            C[i*n+j] =s;
        }
}

void mul_mat_vec(const double* A,const double* x,double* y,int n,int m) {
    for (int i=0;i<n;i++) {
        double s=0.0;
        for(int j =0;j<m;j++)
            s +=A[i*m+j]*x[j];
        y[i] =s;
    }
}

double vector_norm(const double* x,int n) {
    double s = 0.0;
    for(int i=0;i<n; i++) s+=x[i]*x[i];
    return sqrt(s);
}

void normalize(double* x,int n) {
    double norm =vector_norm(x,n);
    if(norm<1e-12) norm =1e-12;
    for(int i=0;i<n;i++) x[i]/= norm;
}

void set_col(double* A,int col,const double* v,int n,int m) {
    for(int i=0;i<n;i++) A[i*m+col]=v[i];
}

void outer_product(const double* a,double* result,int n) {
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
           result[i*n+j]=a[i]*a[j];
}


void scalar_multiply(double* A,double scalar,int size) {
    for(int i=0;i<size;i++) A[i]*=scalar;
}


void MAX_lambda(double* B,int n,double* v_out,double* lambda_max) {
    double* v =(double*)malloc(n * sizeof(double));
    double* v_new= (double*)malloc(n * sizeof(double));

    for(int i=0;i<n;i++) v[i]=(double)rand()/RAND_MAX-0.5;//we dont want all entries of v to be 0 so sub 0.5
    normalize(v,n);

    for(int n_iter=0;n_iter<25;n_iter++) {
        mul_mat_vec(B,v,v_new,n,n);
        normalize(v_new,n);
        copy(v,v_new,n);
    }

    mul_mat_vec(B,v,v_new,n,n);
    *lambda_max= 0.0;
    for(int i=0;i<n;i++) *lambda_max +=v[i]* v_new[i];// this is vt*B*v/vt*v this is called reighleigh quotient

    copy(v_out,v,n);
    free(v);
    free(v_new);
}


void SVD_truncated(double* A,int n,int m,int k,double* Uk,double* Sk,double* Vk) {
    double* At=(double*)malloc(n*m *sizeof(double));
    transpose(A,At,n,m);

    double* B = (double*)malloc(m*m*sizeof(double));
    mul_mat(At,A,B,m,n,m);

    double* temp = (double*)malloc(m*m *sizeof(double));
    copy(temp,B,m*m);

    double* v =  (double*)malloc(m * sizeof(double));
    double* u =  (double*)malloc(n * sizeof(double));
    double* vvt =  (double*)malloc(m*m * sizeof(double));

    for(int i=0;i<k;i++) {
        double lambda;
        MAX_lambda(temp,m,v,&lambda);
        double sigma =sqrt(fmax(0.0, lambda));//this is so that the matrix does not become non positive definite as the sigma value might become less than 0
        Sk[i] = sigma;

        set_col(Vk,i,v,m,k);

        mul_mat_vec(A,v,u,n,m);
        if(sigma < 1e-12) sigma=1e-12;
        scalar_multiply(u,1.0/sigma,n);
        set_col(Uk,i,u,n,k);

        outer_product(v,vvt,m);
        scalar_multiply(vvt,lambda,m * m);
        for(int i=0;i<m*m;i++) temp[i] -= vvt[i];
    }

    free(At);
    free(B);
    free(temp);
    free(v);
    free(u);
    free(vvt);
}


void reconstruct_image(double* Uk,double*Sk,double*Vk,int n,int m,int k,double* output) {
    double* Sk_diag=(double*)malloc(k*k *sizeof(double));
    for(int i=0;i<k*k;i++) Sk_diag[i] = 0.0;
    for(int i=0;i< k;i++) Sk_diag[i*k+i] = Sk[i];

    double* UkSk=(double*)malloc(n*k* sizeof(double));
    double* VkT=(double*)malloc(k*m* sizeof(double));

    mul_mat(Uk,Sk_diag,UkSk,n,k,k);
    transpose(Vk,VkT,m,k);
    mul_mat(UkSk,VkT,output,n,k,m);

    free(Sk_diag);
    free(UkSk);
    free(VkT);
}


void frobenius_norm(double* A,double* B,int size) {
    double sum =0.0;
    for(int i=0;i<size;i++) {
        double diff=A[i]-B[i];
        sum+=pow(diff, 2);
    }
    printf("Frobenius norm: %lf\n",sqrt(sum)*255.0);
}


int main() {
    srand(time(NULL));

    char input_file[100], output_file[100], file_type[10];
    int k;

    printf("Enter type of file (jpg/png): ");
    scanf("%s",file_type);
    printf("Enter input filename: ");
    scanf("%s",input_file);
    printf("Enter output filename: ");
    scanf("%s",output_file);
    printf("Enter rank (k): ");
    scanf("%d",&k);

    int width,height;
    double* input_matrix= load(input_file,&height,&width);

    int n=height,m=width;
    printf("Loaded image: %s (%dx%d)\n",input_file,width,height);

    double* input_copy = (double*)malloc(n*m* sizeof(double));
    copy(input_copy,input_matrix,n*m);

    // printf("Input Matrix:\n");

    // for(int i=0;i<n;i++){
    //     for(int j=0;j<m;j++){
    //         printf("%lf ",input_copy[i*m+j]);
    //     }
    //     printf("\n");
    // }


    double* Uk =(double*)malloc(n*k * sizeof(double));
    double* Sk =(double*)malloc(k * sizeof(double));
    double* Vk =(double*)malloc(m*k * sizeof(double));
    double* output_matrix =(double*)malloc(n*m *sizeof(double));

    SVD_truncated(input_matrix,n,m,k,Uk,Sk,Vk);
    printf("SVD truncated done\n");
    reconstruct_image(Uk,Sk,Vk,n,m,k,output_matrix);
    printf("reconstruct image done\n");

    // printf("Output Matrix:\n");

    // for(int i=0;i<n;i++){
    //     for(int j=0;j<m;j++){
    //         printf("%lf ",output_matrix[i*m+j]);
    //     }
    //     printf("\n");
    // }

    if(strcmp(file_type,"jpg")==0)
        write_jpg(output_file,output_matrix,m,n);
    else
        write_png(output_file,output_matrix,m,n);

    printf("Written to %s\n",output_file);

    frobenius_norm(input_copy,output_matrix,n*m);

    free(input_matrix);
    free(input_copy);
    free(Uk);
    free(Sk);
    free(Vk);
    free(output_matrix);

    return 0;
}
