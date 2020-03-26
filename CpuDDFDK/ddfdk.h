#pragma once

/** 
 * CPU 版本 Distance-Driven 反投影算法
 * 基于 rtk::BackProjectionImageFilter
 */

#include "rtkBackProjectionImageFilter.h"
#include "iostream"
#include <windows.h>
#include <CMath>

template <class TInputImage, class TOutputImage>
class ITK_EXPORT DDBackProjectionImageFilter : public rtk::BackProjectionImageFilter<TInputImage, TOutputImage>
{
  public:
    typedef DDBackProjectionImageFilter                                     Self;
    typedef rtk::BackProjectionImageFilter<TInputImage, TOutputImage>       Superclass;
    typedef itk::SmartPointer<Self>                                         Pointer;
    typedef itk::SmartPointer<const Self>                                   ConstPointer;
    typedef itk::ThreadIdType                                               ThreadIdType;

    typedef typename Superclass::ProjectionMatrixType                       ProjectionMatrixType;
    typedef typename TOutputImage::RegionType                               OutputImageRegionType;
    typedef typename Superclass::ProjectionImageType                        ProjectionImageType;
    typedef typename ProjectionImageType::Pointer                           ProjectionImagePointer;

    itkNewMacro(Self);
    itkTypeMacro(DDBackProjectionImageFilter, ImageToImageFilter);

  protected:
    DDBackProjectionImageFilter() {};
    ~DDBackProjectionImageFilter(){};

    void GenerateOutputInformation() ITK_OVERRIDE;
    //是否真的能够提速？
    //采用预计算的边界坐标，同时可能导致不同线程并发读的问题，需不需要加锁？
    //更新：暂时决定不使用它
    //void PreCalcBorderCoord();
    void ThreadedGenerateData(const OutputImageRegionType &outputRegionForThread, ThreadIdType threadId) ITK_OVERRIDE;
    
    /** Optimized version when the rotation is parallel to X, i.e. matrix[1][0]
    and matrix[2][0] are zeros. */
    void OptimizedBackprojectionX(const OutputImageRegionType& region, const ProjectionMatrixType& matrix,
        const ProjectionImagePointer projection);

    /** Optimized version when the rotation is parallel to Y, i.e. matrix[1][1]
    and matrix[2][1] are zeros. */
    void OptimizedBackprojectionY(const OutputImageRegionType& region, const ProjectionMatrixType& matrix,
        const ProjectionImagePointer projection);

  private:
    DDBackProjectionImageFilter(const Self &);
    void operator = (const Self &);
};

/******实现********/

template<class TInputImage, class TOutputImage>
void DDBackProjectionImageFilter<TInputImage, TOutputImage>
::OptimizedBackprojectionX(const OutputImageRegionType & region, const ProjectionMatrixType & matrix, const ProjectionImagePointer projection)
{
    ////本程序用不到
    ////类似下面的函数
}

template<class TInputImage, class TOutputImage>
void DDBackProjectionImageFilter<TInputImage, TOutputImage>
::OptimizedBackprojectionY(const OutputImageRegionType & region, const ProjectionMatrixType & matrix, const ProjectionImagePointer projection)
{
    typename ProjectionImageType::SizeType pSize = projection->GetBufferedRegion().GetSize();
    typename ProjectionImageType::IndexType pIndex = projection->GetBufferedRegion().GetIndex();
    typename TOutputImage::SizeType vBufferSize = this->GetOutput()->GetBufferedRegion().GetSize();
    typename TOutputImage::IndexType vBufferIndex = this->GetOutput()->GetBufferedRegion().GetIndex();
    typename TInputImage::PixelType *pProj;
    typename TOutputImage::PixelType *pVol, *pVolZeroPointer;

    //pVolZeroPointer 指向 voxel(0,0,0)，并不一定在内存中真实存在
    pVolZeroPointer = this->GetOutput()->GetBufferPointer();
    pVolZeroPointer -= vBufferIndex[0] + vBufferSize[0] * (vBufferIndex[1] + vBufferSize[1] * vBufferIndex[2]);

    //思路：
    //当旋转是绕 Y 时，volume 中沿 Y 方向的 voxel 的投影点 u 坐标有优秀的特质
    //即当体素 j++ 时，投影点 v 不变，u+=du
    //因此在这种情况下，抛弃迭代器，换一种遍历体素的方法（就像下面这样），可以显著降低矩阵乘法的次数

    for (double k = region.GetIndex(2); k < region.GetIndex(2) + region.GetSize(2); k++)
    {
        for (double i = region.GetIndex(0); i < region.GetIndex(0) + region.GetSize(0); i++)
        {
            double j = region.GetIndex(1);

            //i,j,k 为体素 index
            double  w;
            //w 为反投影权重，对每个反投影位置(i,j,k)，w 只与 (i,j,k) 有关
            w = matrix[2][0] * i + matrix[2][2] * k + matrix[2][3];
            w = 1 / w;

            double BorderPoints[8][2]; 

            //BorderPoints 表示如图的 8 个点的 x 与 z 坐标

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

            BorderPoints[0][0] = i;         BorderPoints[0][1] = k - 0.5;
            BorderPoints[1][0] = i - 0.5;   BorderPoints[1][1] = k;
            BorderPoints[2][0] = i + 0.5;   BorderPoints[2][1] = k;
            BorderPoints[3][0] = i;         BorderPoints[3][1] = k + 0.5;
            BorderPoints[4][0] = i - 0.5;   BorderPoints[4][1] = k - 0.5;
            BorderPoints[5][0] = i + 0.5;   BorderPoints[5][1] = k - 0.5;
            BorderPoints[6][0] = i - 0.5;   BorderPoints[6][1] = k + 0.5;
            BorderPoints[7][0] = i + 0.5;   BorderPoints[7][1] = k + 0.5;

            //BorderP[0-9][u,v,du]
            double BorderP[8][3];
            double temp;
            for (int t = 0; t < 8; t++)
            {
                BorderP[t][0] = matrix[0][0] * BorderPoints[t][0] + matrix[0][1] * j + matrix[0][2] * BorderPoints[t][1] + matrix[0][3];
                BorderP[t][1] = matrix[1][0] * BorderPoints[t][0] +                    matrix[1][2] * BorderPoints[t][1] + matrix[1][3];
                temp          = matrix[2][0] * BorderPoints[t][0] +                    matrix[2][2] * BorderPoints[t][1] + matrix[2][3];
                BorderP[t][0] = BorderP[t][0] / temp - pIndex[0];
                BorderP[t][1] = BorderP[t][1] / temp - pIndex[1];
                BorderP[t][2] = matrix[0][1] / temp;
            }

            //pVol 是初始反投影位置
            pVol = pVolZeroPointer + int(i + vBufferSize[0] * (j + k*vBufferSize[1]));
            pProj = projection->GetBufferPointer();

            //预先计算 v 方向的边界列表
            //外接矩形
            double minv = pSize[1], maxv = -1;
            for (int t = 0; t < 8; t++)
            {
                if (BorderP[t][1] < minv) minv = BorderP[t][1];
                if (BorderP[t][1] > maxv) maxv = BorderP[t][1];
            }

            //外接矩形整个出界
            if (minv > pSize[1]-1 || maxv < 0)
                continue;

            //规范化外接矩形
            minv = std::max(minv, 0.);
            maxv = std::min(maxv, double(pSize[1])-1);

            int sizeYList = 0;
            double YList[10];
            YList[0] = minv;
            sizeYList++;

            //投影在沿 v 方向是否退化
            bool flagv = false; 
            if(fabs(maxv-minv)<1e-5)
            {
                YList[1] = maxv;
                sizeYList = 2;
                flagv = true;
            }
            else
            {
                for (double t = std::floor(minv); t < maxv; t++)
                {
                    if (t + 0.5001<maxv && t + 0.5001>minv)
                    {
                        YList[sizeYList] = t + 0.5001; //方便使用 std::round()
                        sizeYList++;
                    }
                }
                YList[sizeYList] = maxv;
                sizeYList++;
            }
            
            int ui, vi;
            for (; j < (region.GetIndex(1) + region.GetSize(1)); j++, pVol += vBufferSize[0])
            {
                //投影在沿 v 方向是否退化
                bool flagu = false;
                
                //在这个循环遍历 voxel，方向是沿 Y 方向  

                //外接矩形
                double maxu = -1, minu = pSize[0];
                for (int t = 0; t < 4; t++)
                {
                    if (BorderP[t][0] + BorderP[t][2] / 2 < minu) minu = BorderP[t][0] + BorderP[t][2] / 2;
                    if (BorderP[t][0] - BorderP[t][2] / 2 < minu) minu = BorderP[t][0] + BorderP[t][2] / 2;
                    if (BorderP[t][0] + BorderP[t][2] / 2 > maxu) maxu = BorderP[t][0] + BorderP[t][2] / 2;
                    if (BorderP[t][0] - BorderP[t][2] / 2 > maxu) maxu = BorderP[t][0] + BorderP[t][2] / 2;
                }
                for (int t = 4; t < 8; t++)
                {
                    if (BorderP[t][0] < minu) minu = BorderP[t][0];
                    if (BorderP[t][0] > maxu) maxu = BorderP[t][0];
                }

                //外接矩形整个出界，跳至下一个 voxel
                if (minu > pSize[0]-1 || maxu < 0)
                {
                    for (int t = 0; t < 8; t++)
                        BorderP[t][0] += BorderP[t][2];
                    continue;
                }

                //规范化外接矩形
                minu = std::max(minu, 0.);
                maxu = std::min(maxu, double(pSize[0])-1);

                //边界点列表
                int sizeXList = 0;
                double XList[10];
                XList[0] = minu;
                sizeXList++;

                if (fabs(maxu - minu) < 1e-5)
                {
                    XList[1] = maxu;
                    sizeXList = 2;
                    flagu = true;
                }
                else
                {
                    for (double t = std::floor(minu); t < maxu; t++)
                    {
                        if (t + 0.5001<maxu && t + 0.5001>minu)
                        {
                            XList[sizeXList] = t + 0.5001; //方便使用 std::round()
                            sizeXList++;
                        }
                    }
                    XList[sizeXList] = maxu;
                    sizeXList++;
                }

                //反投影
                double tv = 0, area = 0, areasum = 0;
                for (int idxx = 0; idxx < sizeXList - 1; idxx++)
                {
                    for (int idxy = 0; idxy < sizeYList - 1; idxy++)
                    {
                        ui = int(std::round(XList[idxx]));
                        vi = int(std::round(YList[idxy]));

                        if (flagu && !flagv)
                            area = YList[idxy + 1] - YList[idxy];
                        else if (!flagu && flagv)
                            area = XList[idxx + 1] - XList[idxx];
                        else if (flagu && flagv)
                            area = 1;
                        else
                            area = (XList[idxx + 1] - XList[idxx])*(YList[idxy + 1] - YList[idxy]);

                        tv += area * *(pProj + pSize[0] * vi + ui);
                        areasum += area;
                    }
                }
                tv /= areasum;

                /*if (std::isnan(tv) || std::isinf(tv))
                {
                    std::cout << "Bad point at: (" << i << j << k << "), you should start debugging!\n";
                    exit(0);
                }*/

                *pVol += w * w * tv;

                for (int t = 0; t < 8; t++) //下一 voxel
                    BorderP[t][0] += BorderP[t][2];
            }

        }
    }

}

template <class TInputImage, class TOutputImage>
void DDBackProjectionImageFilter<TInputImage, TOutputImage>
::GenerateOutputInformation()
{
    if (this->m_Geometry->GetRadiusCylindricalDetector() != 0)
    {
        std::cout << "目前不能處理弧形探測器.";
        exit(0);
    }
    Superclass::GenerateOutputInformation();
}

template <class TInputImage, class TOutputImage>
void DDBackProjectionImageFilter<TInputImage, TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType &outputRegionForThread,
    ThreadIdType threadId)
{
    const unsigned int Dimension = TInputImage::ImageDimension;
    const unsigned int nProj = this->GetInput(1)->GetLargestPossibleRegion().GetSize(Dimension - 1);
    const unsigned int iFirstProj = this->GetInput(1)->GetLargestPossibleRegion().GetIndex(Dimension - 1);

    //迭代器
    typedef itk::ImageRegionConstIterator<TInputImage> InputRegionIterator;
    InputRegionIterator itIn(this->GetInput(), outputRegionForThread);
    typedef itk::ImageRegionIteratorWithIndex<TOutputImage> OutputRegionIterator;
    OutputRegionIterator itOut(this->GetOutput(), outputRegionForThread);

    //初始化
    if (this->GetInput() != this->GetOutput())
    {
        itIn.GoToBegin();
        while (!itIn.IsAtEnd())
        {
            itOut.Set(itOut.Get());
            itOut.Set(0);
            ++itIn;
            ++itOut;
        }
    }


    //旋转中心，假设为 0
    typename TInputImage::PointType rotCenterPoint;
    rotCenterPoint.Fill(0.0);
    itk::ContinuousIndex<double, Dimension> rotCenterIndex;
    this->GetOutput(0)->TransformPhysicalPointToContinuousIndex(rotCenterPoint, rotCenterIndex);

    unsigned int sizex, sizey, sizez;
    sizex = this->GetOutput()->GetLargestPossibleRegion().GetSize(0);
    sizey = this->GetOutput()->GetLargestPossibleRegion().GetSize(1);
    sizez = this->GetOutput()->GetLargestPossibleRegion().GetSize(2);

    LARGE_INTEGER f, end, start;
    QueryPerformanceFrequency(&f);

    for (unsigned int iProj = iFirstProj; iProj < iFirstProj + nProj; iProj++)
    {
        QueryPerformanceCounter(&start);
        //获得当前投影
        ProjectionImagePointer projection;
        projection = this->template GetProjection<ProjectionImageType>(iProj);

        //获得当前投影矩阵
        //采用 index to index 可以降低编程难度，因为 voxel 中心与探测器中心 index 都是整数
        //在本例（Sheep-Logan）中不会影响最后结果，因为 x、y、z 方向的 spacing 都是一致的
        //spacing 不一致时，要使用真实的距离来加权
        ProjectionMatrixType matrix = this->GetIndexToIndexProjectionMatrix(iProj);
        //归一化变换矩阵，保证在 isocenter(0,0,0) 处反投影权重为 1
        double perspFactor = matrix[Dimension - 1][Dimension];
        for (unsigned j = 0; j < Dimension; j++)
            perspFactor += matrix[Dimension - 1][j] * rotCenterIndex[j];
        matrix /= perspFactor;

        //优化版本
        //if (fabs(matrix[1][0])<1e-10 && fabs(matrix[2][0])<1e-10)
        //{
        //    OptimizedBackprojectionX(outputRegionForThread, matrix, projection);

        //    QueryPerformanceCounter(&end);
        //    //if (threadId == 0) {
        //    std::cout << "thread " << threadId << " finished processing projection " << iProj << ' ';
        //    std::cout << "which cost "
        //        << (end.QuadPart - start.QuadPart)
        //        << " units, i.e. " << (end.QuadPart - start.QuadPart)*1.0 / f.QuadPart << " sec" << std::endl;
        //    continue;
        //}

        if (fabs(matrix[1][1])<1e-10 && fabs(matrix[2][1])<1e-10)
        {
            //重点！ 
            OptimizedBackprojectionY(outputRegionForThread, matrix, projection);
            QueryPerformanceCounter(&end);
            
            std::cout << "thread " << threadId << " finished processing projection " << iProj << ' ';
            std::cout << "which cost "
                << (end.QuadPart - start.QuadPart)
                << " units, i.e. " << (end.QuadPart - start.QuadPart)*1.0 / f.QuadPart << " sec" << std::endl;
            continue;
        }


        /*********本程序用不到下面的代碼**********/

        //遍历所有 voxel
        itOut.GoToBegin();
        //预先声明的变量， 提高访问速度
        itk::Index<Dimension - 1> pointProj;
        double ProjSpacingX = projection->GetSpacing()[0];
        double ProjSpacingY = projection->GetSpacing()[1];
        double ProjOriginX = projection->GetOrigin()[0];
        double ProjOriginY = projection->GetOrigin()[0];
        double ProjSizeX = projection->GetLargestPossibleRegion().GetSize(0);
        double ProjSizeY = projection->GetLargestPossibleRegion().GetSize(1); 

        while (!itOut.IsAtEnd())
        {
            //未实现
            //实现方法參考上面的优化版本
            ++itOut;
        }
        QueryPerformanceCounter(&end);
        std::cout << "thread " << threadId << " is processing projection " << iProj << ' ';
        std::cout << "which cost "
            << (end.QuadPart - start.QuadPart)
            << " units, i.e. " << (end.QuadPart - start.QuadPart)*1.0 / f.QuadPart << " sec" << std::endl;
    }
}
