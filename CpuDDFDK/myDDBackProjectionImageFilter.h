#pragma once
/*
* DDBackProjectionImageFilter.h
* CPU版本
* by Angela
* 5/14/2018
*/
#include "rtkBackProjectionImageFilter.h"
#include "iostream"
#include <cmath>
#include <fstream>

#define LINUX

using namespace std;
/*宏定义 计算边界点的投影值*/
#define GET_BORDER_PROJ_MIN_MAX(OFFSETX,OFFSETY,OFFSETZ) \
    tu = matrix[0][0] * (x + OFFSETX) + matrix[0][1] * (y + OFFSETY) + matrix[0][2] * (z + OFFSETZ) + matrix[0][3];\
    tv = matrix[1][0] * (x + OFFSETX) + matrix[1][1] * (y + OFFSETY) + matrix[1][2] * (z + OFFSETZ) + matrix[1][3];\
    tw = matrix[2][0] * (x + OFFSETX) + matrix[2][1] * (y + OFFSETY) + matrix[2][2] * (z + OFFSETZ) + matrix[2][3];\
    tu /= tw; tv /= tw;\
    maxu = (maxu>tu) ? maxu : tu; minu = (minu<tu) ? minu : tu;\
    maxv = (maxv>tv) ? maxv : tv; minv = (minv<tv) ? minv : tv
   
template <class TInputImage, class TOutputImage>
class ITK_EXPORT DDBackProjectionImageFilter : public rtk::BackProjectionImageFilter<TInputImage, TOutputImage> 
{  public:
    typedef DDBackProjectionImageFilter                                     Self;
    typedef rtk::BackProjectionImageFilter<TInputImage, TOutputImage>       Superclass;
    typedef itk::SmartPointer<Self>                                         Pointer;
    typedef itk::SmartPointer<const Self>                                   ConstPointer;
    //typedef itk::ThreadIdType                                               ThreadIdType;

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




template<class TInputImage, class TOutputImage>
void DDBackProjectionImageFilter<TInputImage, TOutputImage>
::OptimizedBackprojectionX(const OutputImageRegionType & region, const ProjectionMatrixType & matrix, const ProjectionImagePointer projection)
{

}

template<class TInputImage, class TOutputImage>
void DDBackProjectionImageFilter<TInputImage, TOutputImage>
::OptimizedBackprojectionY(const OutputImageRegionType & region, const ProjectionMatrixType & matrix, const ProjectionImagePointer projection)
{
   
}

template <class TInputImage, class TOutputImage>
void DDBackProjectionImageFilter<TInputImage, TOutputImage>
::GenerateOutputInformation()
{
    if (this->m_Geometry->GetRadiusCylindricalDetector() != 0)
    {
        std::cout << "目前不能处理弧形探测器。";
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


    //旋转中心，假设为0
    typename TInputImage::PointType rotCenterPoint;
    rotCenterPoint.Fill(0.0);
    itk::ContinuousIndex<double, Dimension> rotCenterIndex;
    this->GetOutput(0)->TransformPhysicalPointToContinuousIndex(rotCenterPoint, rotCenterIndex);
	
    unsigned int sizex, sizey, sizez;
    sizex = this->GetOutput()->GetLargestPossibleRegion().GetSize(0);
    sizey = this->GetOutput()->GetLargestPossibleRegion().GetSize(1);
    sizez = this->GetOutput()->GetLargestPossibleRegion().GetSize(2);

    for (unsigned int iProj = iFirstProj; iProj < iFirstProj + nProj; iProj++)
    {
		// Extract the current slice
        ProjectionImagePointer projection;
        projection = this->template GetProjection<ProjectionImageType>(iProj);

		// Index to index matrix normalized to have a correct backprojection weight
        ProjectionMatrixType matrix = this->GetIndexToIndexProjectionMatrix(iProj);      
        double perspFactor = matrix[Dimension - 1][Dimension];
        for (unsigned j = 0; j < Dimension; j++)
            perspFactor += matrix[Dimension - 1][j] * rotCenterIndex[j];
        matrix /= perspFactor;
        
        //优化版本
        if (fabs(matrix[1][0])<1e-10 && fabs(matrix[2][0])<1e-10)
        {
            continue;
        }

        if (fabs(matrix[1][1])<1e-10 && fabs(matrix[2][1])<1e-10)
        {
            //重点！
            //OptimizedBackprojectionY(outputRegionForThread, matrix, projection);
            continue;
        }

		// Go over each voxel
        itOut.GoToBegin();
       
		//预先声明变量，提高访存速度
        itk::Index<Dimension - 1> pointProj;
        double ProjSpacingX = projection->GetSpacing()[0];
        double ProjSpacingY = projection->GetSpacing()[1];
        double ProjOriginX = projection->GetOrigin()[0];
        double ProjOriginY = projection->GetOrigin()[0];
        double ProjSizeX = projection->GetLargestPossibleRegion().GetSize(0);
        double ProjSizeY = projection->GetLargestPossibleRegion().GetSize(1); 
		/*create 计算投影所用变量*/
        double tu,tv,tw;
        while (!itOut.IsAtEnd())
        {
            int x=itOut.GetIndex()[0],y=itOut.GetIndex()[1],z=itOut.GetIndex()[2];
			//create min max u v 并初始化， 外移提速
            double minu=ProjSizeX,maxu=-1,minv=ProjSizeY,maxv=-1;
			
			/*不用创建数组， 因为只用了一次，即算了一遍之后顺便就把外界矩形算出来了。
			所以，物理意义上的先...然后...在计算机中实现时或许可以穿插执行同时结束*/
            GET_BORDER_PROJ_MIN_MAX(-0.5, 0, -0.5);
            GET_BORDER_PROJ_MIN_MAX(-0.5, 0, 0.5);
            GET_BORDER_PROJ_MIN_MAX(-0.5, -0.5, 0);
            GET_BORDER_PROJ_MIN_MAX(-0.5, 0.5, 0);
            GET_BORDER_PROJ_MIN_MAX(0.5, 0, -0.5);
            GET_BORDER_PROJ_MIN_MAX(0.5, 0, 0.5);
            GET_BORDER_PROJ_MIN_MAX(0.5, -0.5, 0);
            GET_BORDER_PROJ_MIN_MAX(0.5, 0.5, 0);
            GET_BORDER_PROJ_MIN_MAX(0, 0.5, 0.5);
            GET_BORDER_PROJ_MIN_MAX(0, 0.5, -0.5);
            GET_BORDER_PROJ_MIN_MAX(0, -0.5, 0.5);
            GET_BORDER_PROJ_MIN_MAX(0, -0.5, -0.5);
            
			/*防出界处理：1.部分出界 2.整个都出界*/  
			//step1：左边界>=0 右边界<=max
            minv = std::max(minv, 0.);
            maxv = std::min(maxv, double(ProjSizeY - 1));
            minu = std::max(minu, 0.);
            maxu = std::min(maxu, double(ProjSizeX - 1));
            
            //step2: 当左边界>max 或 右边界<0  说明out of projecction range
            if (minu > ProjSizeX || minv > ProjSizeY - 1 || maxu < 0 || maxv < 0)
            {
                ++itOut;
                continue;
            }
			/*反投影权重*/
            tw = matrix[2][0] * (x) + matrix[2][1] * (y) + matrix[2][2] * (z) + matrix[2][3];
            tw=1/tw;
            tw*=tw;
            
            
            bool flagu = false, flagv = false;
            if (maxu - minu < 1e-5)
                flagu = true;
            if (maxv - minv < 1e-5)
                flagv = true;

            itk::Index<2> projIndex;
                        
            double voxel_data = 0;
            /*正常情况*/
			if (!flagu && !flagv)
            {
                double sumarea = 0., area, du;
                for (tu = floor(minu + 0.5); tu <= ceil(maxu - 0.5); tu++)
                {
                    du = min(maxu, tu + 0.5) - max(minu, tu - 0.5);
					//不用list来存
                    for (tv = floor(minv + 0.5); tv <= ceil(maxv - 0.5); tv++)
                    {
                        area = du * (min(maxv, tv + 0.5) - max(minv, tv - 0.5));
                        sumarea += area;/*用于权重*/

                        projIndex[0]=int(tu);projIndex[1]=int(tv);/*标记要取值的投影点位置*/

                        voxel_data += area * projection->GetPixel(projIndex);
                    }
                }
                voxel_data /= sumarea;
            }
			/*u方向退化*/
            else if (flagu && !flagv)
            {
                tu = round((maxu + minu) / 2);/* round(x)返回x的四舍五入整数值*/
                double sumarea = 0., area;//【】
                for (tv = floor(minv + 0.5); tv <= ceil(maxv - 0.5); tv++)
                {
                    area = min(maxv, tv + 0.5) - max(minv, tv - 0.5);
                    sumarea += area;
                    projIndex[0]=int(tu);projIndex[1]=int(tv);

                    voxel_data += area * projection->GetPixel(projIndex);
                }
                voxel_data /= sumarea;
            }
			/*v方向退化*/
            else if (!flagu && flagv)
            {
                tv = round((maxv + minv) / 2);
                double sumarea = 0., area;
                for (tu = floor(minu + 0.5); tu <= ceil(maxu - 0.5); tu++)
                {
                    area = min(maxu, tu + 0.5) - max(minu, tu - 0.5);
                    sumarea += area;
                    projIndex[0]=int(tu);projIndex[1]=int(tv);

                    voxel_data += area * projection->GetPixel(projIndex);
                }
                voxel_data /= sumarea;
            }
			/*u,v都退化*/
            else
            {
                projIndex[0]=int(round(minu));projIndex[1]=int(round(minv));

                voxel_data += projection->GetPixel(projIndex);
            }

            itOut.Set(itOut.Get()+tw*voxel_data);/*tw=1/(w^2)*/

            ++itOut;
        }
    }
}
