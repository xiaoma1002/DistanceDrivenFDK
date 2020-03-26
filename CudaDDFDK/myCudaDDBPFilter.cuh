#ifndef myCudaDDBPFilter_cuh
#define myCudaDDBPFilter_cuh

#define NumProjPerProcess 15

#define P2Z 1
#define P2X 0

#define WINDOWS

#define CHECK_SOME_VALUE(I,J,K,exp)\
    if(i==I&&j==J&&k==K)\
    {\
        printf(#exp": %f \n", exp);\
    }


#define GET_FIRST_PROJ_POINTS(OFFSETX,OFFSETY,OFFSETZ,n)                                    \
    BorderP[n][0] = m00 * (i + OFFSETX) + m01 * (j + OFFSETY) + m02 * (k + OFFSETZ) + m03;  \
    BorderP[n][1] = m10 * (i + OFFSETX) + m11 * (j + OFFSETY) + m12 * (k + OFFSETZ) + m13;  \
    tw            = m20 * (i + OFFSETX) + m21 * (j + OFFSETY) + m22 * (k + OFFSETZ) + m23;  \
    BorderP[n][0] /= tw; BorderP[n][1] /= tw;                                               \
    BorderP[n][2] = m01 / tw;                                                               \
    maxv = (maxv>BorderP[n][1]) ? maxv : BorderP[n][1];                                     \
    minv = (minv<BorderP[n][1]) ? minv : BorderP[n][1]

#define GET_FIRST_PROJ_POINTS_C(OFFSETX,OFFSETY,OFFSETZ,n)                                    \
    BorderP[n][0] = c_matrices[m_offset + 0] * (i + OFFSETX) + c_matrices[m_offset + 1] * (j + OFFSETY) + c_matrices[m_offset + 2] * (k + OFFSETZ) + c_matrices[m_offset + 3];\
    BorderP[n][1] = c_matrices[m_offset + 4] * (i + OFFSETX) + c_matrices[m_offset + 5] * (j + OFFSETY) + c_matrices[m_offset + 6] * (k + OFFSETZ) + c_matrices[m_offset + 7];\
    tw            = c_matrices[m_offset + 8] * (i + OFFSETX) + c_matrices[m_offset + 9] * (j + OFFSETY) + c_matrices[m_offset + 10] * (k + OFFSETZ) + c_matrices[m_offset + 11];\
    BorderP[n][0] /= tw; BorderP[n][1] /= tw;                                               \
    BorderP[n][2] = c_matrices[m_offset + 1] / tw;                                          \
    maxv = (maxv>BorderP[n][1]) ? maxv : BorderP[n][1];                                     \
    minv = (minv<BorderP[n][1]) ? minv : BorderP[n][1]

#define LOAD_MAT_C \
    int m_offset = iProj * 12;\
    float   m00 = c_matrices[m_offset + 0], m01 = c_matrices[m_offset + 1], m02 = c_matrices[m_offset + 2], m03 = c_matrices[m_offset + 3],\
            m10 = c_matrices[m_offset + 4], m11 = c_matrices[m_offset + 5], m12 = c_matrices[m_offset + 6], m13 = c_matrices[m_offset + 7],\
            m20 = c_matrices[m_offset + 8], m21 = c_matrices[m_offset + 9], m22 = c_matrices[m_offset + 10], m23 = c_matrices[m_offset + 11]

#define GET_BORDER_PROJ_MIN_MAX(OFFSETX,OFFSETY,OFFSETZ) \
    tu = m00 * (i + OFFSETX) + m01 * (j + OFFSETY) + m02 * (k + OFFSETZ) + m03;\
    tv = m10 * (i + OFFSETX) + m11 * (j + OFFSETY) + m12 * (k + OFFSETZ) + m13;\
    tw = m20 * (i + OFFSETX) + m21 * (j + OFFSETY) + m22 * (k + OFFSETZ) + m23;\
    tu /= tw; tv /= tw;\
    maxu = (maxu>tu) ? maxu : tu; minu = (minu<tu) ? minu : tu;\
    maxv = (maxv>tv) ? maxv : tv; minv = (minv<tv) ? minv : tv

#define GET_BORDER_PROJ_MIN_MAX_C(OFFSETX,OFFSETY,OFFSETZ) \
    tu = c_matrices[m_offset + 0] * (i + OFFSETX) + c_matrices[m_offset + 1] * (j + OFFSETY) + c_matrices[m_offset + 2] * (k + OFFSETZ) + c_matrices[m_offset + 3];\
    tv = c_matrices[m_offset + 4] * (i + OFFSETX) + c_matrices[m_offset + 5] * (j + OFFSETY) + c_matrices[m_offset + 6] * (k + OFFSETZ) + c_matrices[m_offset + 7];\
    tw = c_matrices[m_offset + 8] * (i + OFFSETX) + c_matrices[m_offset + 9] * (j + OFFSETY) + c_matrices[m_offset + 10] * (k + OFFSETZ) + c_matrices[m_offset + 11];\
    tu /= tw; tv /= tw;\
    maxu = (maxu>tu) ? maxu : tu; minu = (minu<tu) ? minu : tu;\
    maxv = (maxv>tv) ? maxv : tv; minv = (minv<tv) ? minv : tv

#define CUDA_TIMER_START(name)\
    cudaEvent_t start##name, stop##name;\
    float elapsedTime##name = 0.0;\
    cudaEventCreate(&start##name);\
    cudaEventCreate(&stop##name);\
    cudaEventRecord(start##name, 0)

#define CUDA_TIMER_STOP(name)\
    cudaEventRecord(stop##name, 0);\
    cudaEventSynchronize(stop##name);\
    cudaEventElapsedTime(&elapsedTime##name, start##name, stop##name);\
    printf(#name" Time: %f sec\n", elapsedTime##name/1000.0);\
    cudaEventDestroy(start##name);\
    cudaEventDestroy(stop##name)

extern "C"
void CUDA_DDBackProjection(int proj_size[3],
    int vol_size[3],
    float*matrices,     //Host
    float*dev_vol_in,   //Device
    float*dev_vol_out,  //Device
    float*dev_proj,
    int *Direction);     //Host

#endif