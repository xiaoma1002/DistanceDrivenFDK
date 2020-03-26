/*****************
*  rtk #includes *
*****************/
#include "rtkCudaUtilities.hcu"
#include "rtkConfiguration.h"
#include "myCudaDDBPFilter.cuh"

/*****************
*  C   #includes *
*****************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*****************
* CUDA #includes *
*****************/
#include <cuda.h>

//Constant Memory
__constant__ float  c_matrices[NumProjPerProcess * 12];
__constant__ int    c_direction[NumProjPerProcess];
__constant__ int3   c_projSize;
__constant__ int3   c_vol_size;


//Texture Memory, to hold projections
texture<float,cudaTextureType2DLayered> tex_proj;

//Optimized Kernel Code
__global__
void kernel_ddfdk_opti(float *dev_vol_in, float * dev_vol_out)
{
    __shared__  float BorderP[8][3]; 
    __shared__  float minv,maxv;

    unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    unsigned int j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    unsigned int k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    //invalid voxel coord
    if (i >= c_vol_size.x - 1 || j >= c_vol_size.y - 1 || k >= c_vol_size.z - 1)
        return;

    //backprojection index
    long int vol_idx = i + (j + k*c_vol_size.y)*(c_vol_size.x);

    //to hold backprojection value
    float voxel = 0;
    
    for (int iProj = 0; iProj < NumProjPerProcess; iProj++)
    {
        __syncthreads();

        
        
        LOAD_MAT_C;

        float tw=0;

            /*
            *    4      0      5
            *    *------*------*         --------> x
            *    |             |         |
            *    |   (i,j,k)   |         |    
            *   1*      *      *2        |
            *    |             |         | z   
            *    |             |             
            *    *------*------*
            *    6      3      7
            */

        //int m_offset = iProj * 12;

        if (j == 0)
        {
            minv = c_projSize.y;
            maxv = -1;

            GET_FIRST_PROJ_POINTS(0, 0, -0.5, 0);
            GET_FIRST_PROJ_POINTS(-0.5, 0, 0, 1);
            GET_FIRST_PROJ_POINTS(0.5, 0, 0, 2);
            GET_FIRST_PROJ_POINTS(0, 0, 0.5, 3);
            GET_FIRST_PROJ_POINTS(-0.5, 0, -0.5, 4);
            GET_FIRST_PROJ_POINTS(0.5, 0, -0.5, 5);
            GET_FIRST_PROJ_POINTS(-0.5, 0, 0.5, 6);
            GET_FIRST_PROJ_POINTS(0.5, 0, 0.5, 7);

            // GET_FIRST_PROJ_POINTS_C(0, 0, -0.5, 0);
            // GET_FIRST_PROJ_POINTS_C(-0.5, 0, 0, 1);
            // GET_FIRST_PROJ_POINTS_C(0.5, 0, 0, 2);
            // GET_FIRST_PROJ_POINTS_C(0, 0, 0.5, 3);
            // GET_FIRST_PROJ_POINTS_C(-0.5, 0, -0.5, 4);
            // GET_FIRST_PROJ_POINTS_C(0.5, 0, -0.5, 5);
            // GET_FIRST_PROJ_POINTS_C(-0.5, 0, 0.5, 6);
            // GET_FIRST_PROJ_POINTS_C(0.5, 0, 0.5, 7);

            //out of range
            if (minv > c_projSize.y - 1 || maxv < 0)
                continue;

            //minv=max(0.,minv);
            //maxv=min(maxv,float(c_projSize.y-1));
        }
      
        //done loading first voxel
        __syncthreads();

        float minu=c_projSize.x,maxu=-1;

        for (unsigned int index = 0; index < 4; index++)
        {
            tw = BorderP[index][0] + float(j + 0.5) * BorderP[index][2];//u+=y*du

            //minu=min(minu,tw);maxu=max(maxu,tw);

             minu = minu < tw ? minu : tw;
             maxu = maxu > tw ? maxu : tw;
            tw = BorderP[index][0] + float(j - 0.5) * BorderP[index][2];

            //minu=min(minu,tw);maxu=max(maxu,tw);
             minu = minu < tw ? minu : tw;
             maxu = maxu > tw ? maxu : tw;
        }
        for (unsigned int index = 4; index < 7; index++)
        {
            tw = BorderP[index][0] + j * BorderP[index][2];

            //minu=min(minu,tw);maxu=max(maxu,tw);

             minu = minu < tw ? minu : tw;
             maxu = maxu > tw ? maxu : tw;
        }

        //out of range
        if(minu>c_projSize.x-1||maxu<0)
            continue;

        //minv=max(0.,minv);
        //maxv=min(maxv,float(c_projSize.y-1));

        //current perspFactor
        tw = m20*i + m21*j + m22*k + m23;
        //tw = c_matrices[m_offset + 8]*i + c_matrices[m_offset + 9]*j + c_matrices[m_offset + 10]*k + c_matrices[m_offset + 11];
        tw = 1 / tw;
        tw *= tw;
                 
        float voxel_data =  tex2DLayered(tex_proj, maxv, maxu, iProj)+
                            tex2DLayered(tex_proj, minv, minu, iProj)-
                            tex2DLayered(tex_proj, minv, maxu, iProj)-
                            tex2DLayered(tex_proj, maxv, minu, iProj);

        
        float area = (maxu-minu)*(maxv-minv);
        
        area = area > 1e-5 ? area:1;
        
        voxel+=voxel_data/area*tw;
    }
    dev_vol_out[vol_idx] = dev_vol_in[vol_idx] + voxel;
}

//Kernel Code
__global__
void kernel_ddfdk(float *dev_vol_in, float * dev_vol_out)
{
    unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    unsigned int j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    unsigned int k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    //invalid voxel coord
    if (i >= c_vol_size.x - 1 || j >= c_vol_size.y - 1 || k >= c_vol_size.z - 1)
        return;

    //backprojection index
    long int vol_idx = i + (j + k*c_vol_size.y)*(c_vol_size.x);

    //to hold backprojection value
    float voxel = 0;
    
    //some temp value
    float tu, tv, tw, minu, maxu, minv, maxv;

    for (int iProj = 0; iProj < NumProjPerProcess; iProj++)
    {
        // load current projection matrix
        LOAD_MAT_C;

        minu = c_projSize.x; maxu = -1; minv = c_projSize.y; maxv = -1;
        
        
        if(c_direction[iProj])
        {
            //find bounding rect
            GET_BORDER_PROJ_MIN_MAX(0.5, 0.5, 0);
            GET_BORDER_PROJ_MIN_MAX(-0.5, 0.5, 0);
            GET_BORDER_PROJ_MIN_MAX(0.5, -0.5, 0);
            GET_BORDER_PROJ_MIN_MAX(-0.5, -0.5, 0);
        }
        else
        {
            //find bounding rect
            GET_BORDER_PROJ_MIN_MAX(0, 0.5, 0.5);
            GET_BORDER_PROJ_MIN_MAX(0, 0.5, -0.5);
            GET_BORDER_PROJ_MIN_MAX(0, -0.5, 0.5);
            GET_BORDER_PROJ_MIN_MAX(0, -0.5, -0.5);           
        }

        // //顶点
        // GET_BORDER_PROJ_MIN_MAX(-0.5, 0.5, 0.5);
        // GET_BORDER_PROJ_MIN_MAX(-0.5, 0.5, -0.5);
        // GET_BORDER_PROJ_MIN_MAX(-0.5, -0.5, 0.5);
        // GET_BORDER_PROJ_MIN_MAX(-0.5, -0.5, -0.5);
        // GET_BORDER_PROJ_MIN_MAX(0.5, 0.5, -0.5);
        // GET_BORDER_PROJ_MIN_MAX(0.5, 0.5, 0.5);
        // GET_BORDER_PROJ_MIN_MAX(0.5, -0.5, -0.5);
        // GET_BORDER_PROJ_MIN_MAX(0.5, -0.5, 0.5);

        // //棱边
        // GET_BORDER_PROJ_MIN_MAX(-0.5, 0, -0.5);
        // GET_BORDER_PROJ_MIN_MAX(-0.5, 0, 0.5);
        // GET_BORDER_PROJ_MIN_MAX(-0.5, -0.5, 0);
        // GET_BORDER_PROJ_MIN_MAX(-0.5, 0.5, 0);
        // GET_BORDER_PROJ_MIN_MAX(0.5, 0, -0.5);
        // GET_BORDER_PROJ_MIN_MAX(0.5, 0, 0.5);
        // GET_BORDER_PROJ_MIN_MAX(0.5, -0.5, 0);
        // GET_BORDER_PROJ_MIN_MAX(0.5, 0.5, 0);
        // GET_BORDER_PROJ_MIN_MAX(0, 0.5, 0.5);
        // GET_BORDER_PROJ_MIN_MAX(0, 0.5, -0.5);
        // GET_BORDER_PROJ_MIN_MAX(0, -0.5, 0.5);
        // GET_BORDER_PROJ_MIN_MAX(0, -0.5, -0.5);

        // //面心
        // GET_BORDER_PROJ_MIN_MAX(-0.5, 0, 0);
        // GET_BORDER_PROJ_MIN_MAX(0.5, 0, 0);
        // GET_BORDER_PROJ_MIN_MAX(0, -0.5, 0);
        // GET_BORDER_PROJ_MIN_MAX(0, 0.5, 0);
        // GET_BORDER_PROJ_MIN_MAX(0, 0, -0.5);
        // GET_BORDER_PROJ_MIN_MAX(0, 0, 0.5);



              
        //out of projecction range
        if (minu > c_projSize.x - 1 || minv > c_projSize.y - 1 || maxu < 0 || maxv < 0)
            continue;
        
        //current perspFactor
        tw = m20*i + m21*j + m22*k + m23;
        tw = 1 / tw;
        tw *= tw;
                 
        float voxel_data =  tex2DLayered(tex_proj, maxv, maxu, iProj)+
                            tex2DLayered(tex_proj, minv, minu, iProj)-
                            tex2DLayered(tex_proj, minv, maxu, iProj)-
                            tex2DLayered(tex_proj, maxv, minu, iProj);

        
        float area = (maxu-minu)*(maxv-minv);
        area = area > 1e-5 ? area:1;
        
        voxel+=voxel_data/area*tw;
    }

    dev_vol_out[vol_idx] = dev_vol_in[vol_idx] + voxel;
}

//Calculate projection integral image
void calcProjInt(float *d_proj,int proj_size[3])
{
    float *h_proj=new float[proj_size[0]*proj_size[1]*proj_size[2]];
    cudaMemcpy(h_proj, d_proj, proj_size[0]*proj_size[1]*proj_size[2]*sizeof(float), cudaMemcpyDeviceToHost);

    for(int iProj=0;iProj<proj_size[2];iProj++)
    {
        float *p_temp=h_proj+iProj*proj_size[0]*proj_size[1];

        for(int i=0;i<proj_size[1];i++)
        {
            for(int j=0;j<proj_size[0];j++)
            {
                if(j!=0) *(p_temp+i*proj_size[0]+j)+=*(p_temp+i*proj_size[0]+j-1);
            }
        }

        for(int j=0;j<proj_size[0];j++)
        {
            for(int i=0;i<proj_size[1];i++)
            {
                if(i!=0) *(p_temp+i*proj_size[0]+j)+=*(p_temp+(i-1)*proj_size[0]+j);
            }
        }
    }

    cudaMemcpy(d_proj, h_proj, proj_size[0]*proj_size[1]*proj_size[2]*sizeof(float), cudaMemcpyHostToDevice);
    delete[]h_proj;
}


void CUDA_DDBackProjection(int proj_size[3],
    int vol_size[3],
    float*matrices,     //Host
    float*dev_vol_in,   //Device
    float*dev_vol_out,  //Device
    float*dev_proj,     //Device
    int *Direction)     //Host
{
    //Copy some data to constant memory
    cudaMemcpyToSymbol(c_direction, &(Direction[0]), sizeof(int)*proj_size[2]);
    cudaMemcpyToSymbol(c_projSize, proj_size, sizeof(int3));
    cudaMemcpyToSymbol(c_vol_size, vol_size, sizeof(int3));
    cudaMemcpyToSymbol(c_matrices, &(matrices[0]), 12 * sizeof(float) *proj_size[2]);
    CUDA_CHECK_ERROR;

    //Calculate projection integral image
    //calcProjInt(dev_proj,proj_size);
       
    //Malloc CUDA array and copy data to hold projections, and use as Texture
    //Malloc
    cudaArray* arr_proj;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent projExtent = make_cudaExtent(proj_size[0], proj_size[1], proj_size[2]);
    cudaMalloc3DArray((cudaArray**)&arr_proj, &channelDesc, projExtent, cudaArrayLayered);
    //Copy
    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcPtr = make_cudaPitchedPtr(dev_proj, proj_size[0] * sizeof(float), proj_size[0], proj_size[1]);
    copyParams.dstArray = (cudaArray*)arr_proj;
    copyParams.extent = projExtent;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copyParams);
    CUDA_CHECK_ERROR;
    //Set and bind texture
    tex_proj.normalized = false;    //no normalize
    tex_proj.filterMode = cudaFilterModeLinear;  
    tex_proj.addressMode[0] = cudaAddressModeClamp; 
    tex_proj.addressMode[1] = cudaAddressModeClamp;
    tex_proj.addressMode[2] = cudaAddressModeClamp;
    cudaBindTextureToArray(tex_proj, (cudaArray*)arr_proj, channelDesc);
    CUDA_CHECK_ERROR;

    //Thread structure and launch kernel
    const int tBlock_x = 1;
    const int tBlock_y = vol_size[1];
    const int tBlock_z = 1;
   
    unsigned int  blocksInX = (vol_size[0] - 1) / tBlock_x + 1;
    unsigned int  blocksInY = (vol_size[1] - 1) / tBlock_y + 1;
    unsigned int  blocksInZ = (vol_size[2] - 1) / tBlock_z + 1;

    dim3 dimGrid = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);
    
    //Launch kernel
    CUDA_TIMER_START(ddfdk);
   // kernel_ddfdk_opti << < dimGrid, dimBlock >> > (dev_vol_in, dev_vol_out);
	kernel_ddfdk << < dimGrid, dimBlock >> > (dev_vol_in, dev_vol_out);
    CUDA_CHECK_ERROR;
    CUDA_TIMER_STOP(ddfdk);

    //Unbind texture and free space
    cudaUnbindTexture(tex_proj);
    cudaFreeArray((cudaArray*)arr_proj);
}