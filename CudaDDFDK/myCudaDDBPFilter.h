/**
* 实现 CUDA 版本 Distance-Driven 反投影算法
* 基于 rtk::FDKBackProjectionImageFilter
*/

#ifndef myCudaDDBPFilter_h
#define myCudaDDBPFilter_h

#include <rtkFDKBackProjectionImageFilter.h>
#include <rtkWin32Header.h>
#include <itkCudaImage.h>
#include <itkCudaInPlaceImageFilter.h>
#include <rtkCudaUtilities.hcu>
#include <itkMacro.h>
#include "myCudaDDBPFilter.cuh"

class  myCudaDDBPFilter :
  public itk::CudaInPlaceImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3>,
  rtk::FDKBackProjectionImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3> > >
{
public:
    /* Standard class typedefs. */
    typedef itk::CudaImage<float, 3>                             ImageType;
    typedef FDKBackProjectionImageFilter< ImageType, ImageType>  FDKBackProjectionImageFilterType;
    typedef myCudaDDBPFilter                                     Self;
    typedef itk::CudaInPlaceImageFilter<ImageType, ImageType,
        FDKBackProjectionImageFilterType>                        Superclass;
    typedef itk::SmartPointer<Self>                              Pointer;
    typedef itk::SmartPointer<const Self>                        ConstPointer;

    typedef ImageType::RegionType            OutputImageRegionType;
    typedef itk::CudaImage<float, 2>         ProjectionImageType;
    typedef ProjectionImageType::Pointer     ProjectionImagePointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(myCudaDDBPFilter, Superclass);

protected:
    myCudaDDBPFilter();
    virtual ~myCudaDDBPFilter() {};

    virtual void GPUGenerateData();

private:
    myCudaDDBPFilter(const Self&);                 //purposely not implemented
    void operator=(const Self&);                   //purposely not implemented

    void PreCalcProjIntegralImage(float *d_Proj,int Proj_Size[3]);
};


/******************** 实现 ************************/

myCudaDDBPFilter::myCudaDDBPFilter(){}

void myCudaDDBPFilter::PreCalcProjIntegralImage(float *d_Proj,int Proj_Size[3])
{
    float *h_Proj=new float[Proj_Size[0]*Proj_Size[1]*Proj_Size[2]];
    cudaMemcpy(h_Proj, d_Proj, Proj_Size[0]*Proj_Size[1]*Proj_Size[2]*sizeof(float), cudaMemcpyDeviceToHost);

    for(int iProj=0;iProj<Proj_Size[2];iProj++)
    {
        float *p_temp=h_Proj+iProj*Proj_Size[0]*Proj_Size[1];

        for(int i=0;i<Proj_Size[1];i++)
        {
            for(int j=0;j<Proj_Size[0];j++)
            {
                if(j!=0) *(p_temp+i*Proj_Size[0]+j)+=*(p_temp+i*Proj_Size[0]+j-1);
            }
        }

        for(int j=0;j<Proj_Size[0];j++)
        {
            for(int i=0;i<Proj_Size[1];i++)
            {
                if(i!=0) *(p_temp+i*Proj_Size[0]+j)+=*(p_temp+(i-1)*Proj_Size[0]+j);
            }
        }
    }

    cudaMemcpy(d_Proj, h_Proj, Proj_Size[0]*Proj_Size[1]*Proj_Size[2]*sizeof(float), cudaMemcpyHostToDevice);
    delete[]h_Proj;
}


void myCudaDDBPFilter::GPUGenerateData()
{
    const unsigned int Dimension = ImageType::ImageDimension;
    const unsigned int nProj = 
        this->GetInput(1)->GetLargestPossibleRegion().GetSize(Dimension - 1);
    const unsigned int iFirstProj = 
        this->GetInput(1)->GetLargestPossibleRegion().GetIndex(Dimension - 1);

    if (nProj > 1024)
        itkGenericExceptionMacro("Too many projecctions\n")

    ImageType::PointType rotCenterPoint;
    rotCenterPoint.Fill(0.0);
    itk::ContinuousIndex<double, Dimension> rotCenterIndex;
    this->GetInput(0)->TransformPhysicalPointToContinuousIndex(rotCenterPoint, rotCenterIndex);

    for (unsigned int i = 0; i < 3; i++)
        rotCenterIndex[i] -= this->GetOutput()->GetRequestedRegion().GetIndex()[i];

    //ProjectionSize[XSize,YSize,num]
    int ProjectionSize[3];
    ProjectionSize[0] = this->GetInput(1)->GetBufferedRegion().GetSize()[0];
    ProjectionSize[1] = this->GetInput(1)->GetBufferedRegion().GetSize()[1];
    ProjectionSize[2] = nProj;

    int volumeSize[3];
    volumeSize[0] = this->GetOutput()->GetBufferedRegion().GetSize()[0];
    volumeSize[1] = this->GetOutput()->GetBufferedRegion().GetSize()[1];
    volumeSize[2] = this->GetOutput()->GetBufferedRegion().GetSize()[2];

    //On device
    float *pin = *(float**)(this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer());
    float *pout = *(float**)(this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer());

    //point to projections
    float *stackGPUPointer = *(float**)(this->GetInput(1)->GetCudaDataManager()->GetGPUBufferPointer());
    //projection size in one dimension
    ptrdiff_t projSize = this->GetInput(1)->GetBufferedRegion().GetSize()[0]*
        this->GetInput(1)->GetBufferedRegion().GetSize()[1];
    stackGPUPointer += projSize*(iFirstProj - this->GetInput(1)->GetBufferedRegion().GetIndex()[2]);

    PreCalcProjIntegralImage(stackGPUPointer,ProjectionSize);
    
    //hold all the projection matrix
    float *fMatrix = new float[12 * nProj];
    int *Direction=new int[nProj];
    this->SetTranspose(true);

    for (unsigned int iProj = iFirstProj; iProj < iFirstProj + nProj; iProj++)
    {
        //归一化的投影矩阵，保证在 isocenter 处反投影权重为 1
        ProjectionMatrixType matrix = GetIndexToIndexProjectionMatrix(iProj);

        double gantryAngle=this->m_Geometry->GetGantryAngles()[iProj-iFirstProj];
        if ((gantryAngle >= 0 && gantryAngle <= 0.7854) || (gantryAngle >= 2.3562 && gantryAngle <= 3.9270) || (gantryAngle >= 5.4978 && gantryAngle < 6.2832))
            *(Direction + iProj) = P2Z;
        else
            *(Direction + iProj) = P2X;

        double perspFactor = matrix[Dimension - 1][Dimension];
        for (unsigned int j = 0; j < Dimension; j++)
            perspFactor += matrix[Dimension - 1][j] * rotCenterIndex[j];
        matrix /= perspFactor;

        //std::swap(matrix[0][0], matrix[0][1]);
        //std::swap(matrix[1][0], matrix[1][1]);

        //一维化存储所有投影矩
        for (int j = 0; j < 12; j++)
            fMatrix[j + (iProj - iFirstProj) * 12] = matrix[j / 4][j % 4];
    }

    for (unsigned int i = 0; i < nProj; i += NumProjPerProcess)
    {
        //NumProjPerProcess 为每次处理的投影数量。与 CUDA 性能有关

        ProjectionSize[2] = std::min(nProj - 1, (unsigned int)NumProjPerProcess);

        CUDA_DDBackProjection(ProjectionSize,
            volumeSize,
            fMatrix + 12 * i,
            pin,
            pout,
            stackGPUPointer + projSize*i,
            Direction+i);

        pin = pout;
    }
    delete[]fMatrix;
    delete[]Direction;
}

#endif