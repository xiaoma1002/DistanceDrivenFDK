#include <itkImage.h>
#include <rtkProjectionsReader.h>
#include <rtkThreeDCircularProjectionGeometryXMLFile.h>
#include <rtkConstantImageSource.h>
#include <rtkCudaFDKWeightProjectionFilter.h>
#include <rtkCudaFFTRampImageFilter.h>
#include <itkStreamingImageFilter.h>
#include <itkImageRegionSplitterDirection.h>
#include <iostream>

#include "myCudaDDBPFilter.h"
#include "myCudaDDBPFilter.cuh"

#define WINDOWS


#ifdef WINDOWS
#include <windows.h>
#else
#include <time.h>
#endif

using namespace std;

#define CHECK_ERROR(operate)\
    try{operate;}\
    catch (itk::ExceptionObject &err) { cerr << "异常!" << endl;cerr << err << endl; return 1;}

int main()
{
#ifdef WINDOWS
    LARGE_INTEGER f, end, start;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&start);
#else
    clock_t start,end;
    start=clock();
#endif
    typedef float OutputPixelType;
    const unsigned int Dimension = 3;
    typedef itk::CudaImage< OutputPixelType, Dimension > OutputImageType;

    //Projection Reader
    typedef rtk::ProjectionsReader< OutputImageType > ReaderType;
    ReaderType::Pointer reader = ReaderType::New();

    std::vector<std::string> filenames;
    filenames.push_back("projections.mha");
    reader->SetFileNames(filenames);
    CHECK_ERROR(reader->UpdateOutputInformation())

    //Geometry
    rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader;
    geometryReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
    geometryReader->SetFilename("geometry.xml");
    CHECK_ERROR(geometryReader->GenerateOutputInformation())

    //重建 Volume：constantImageSource
    typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
    ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
    typedef typename ConstantImageSourceType::OutputImageType ImageType;
    typename ImageType::SizeType imageDimension;
    imageDimension.Fill(256);
    typename ImageType::SpacingType imageSpacing;
    imageSpacing.Fill(2);
    typename ImageType::PointType imageOrigin;
    for (unsigned int i = 0; i<3; i++)
        imageOrigin[i] = imageSpacing[i] * (imageDimension[i] - 1) * -0.5;
    typename ImageType::DirectionType imageDirection;
    imageDirection.SetIdentity();
    constantImageSource->SetOrigin(imageOrigin);
    constantImageSource->SetSpacing(imageSpacing);
    constantImageSource->SetDirection(imageDirection);
    constantImageSource->SetSize(imageDimension);
    constantImageSource->SetConstant(0.);
    CHECK_ERROR(constantImageSource->UpdateOutputInformation())

    //********* FDK 重建 **********//
    typedef itk::ExtractImageFilter<OutputImageType, OutputImageType>   ExtractFilterType;
    typename ExtractFilterType::Pointer                                 ExtractFilter;
    ExtractFilter = ExtractFilterType::New();
    typedef rtk::CudaFDKWeightProjectionFilter                          WeightFilterType;
    typename WeightFilterType::Pointer                                  WeightFilter;
    WeightFilter = WeightFilterType::New();
    typedef rtk::CudaFFTRampImageFilter                                 RampFilterType;
    typename RampFilterType::Pointer                                    RampFilter;
    RampFilter = RampFilterType::New();
    typedef myCudaDDBPFilter                                            BackProjectionFilterType;
    typename BackProjectionFilterType::Pointer                          BackProjectionFilter;
    BackProjectionFilter = BackProjectionFilterType::New();

    BackProjectionFilter->InPlaceOn();

    //加权
    WeightFilter->SetInput(reader->GetOutput());
    WeightFilter->InPlaceOn();
    WeightFilter->SetGeometry(geometryReader->GetOutputObject());

    //滤波
    RampFilter->SetInput(WeightFilter->GetOutput());
    RampFilter->SetTruncationCorrection(0.0);
    RampFilter->SetHannCutFrequency(0.0);
    RampFilter->SetHannCutFrequencyY(0.0);

    //反投
    BackProjectionFilter->SetInput(0, constantImageSource->GetOutput());
    BackProjectionFilter->SetInput(1, RampFilter->GetOutput());
    BackProjectionFilter->SetGeometry(geometryReader->GetOutputObject());

    typedef itk::StreamingImageFilter<OutputImageType, OutputImageType> StreamerType;
    StreamerType::Pointer streamerBP = StreamerType::New();
    streamerBP->SetInput(BackProjectionFilter->GetOutput());
    streamerBP->SetNumberOfStreamDivisions(1);
    itk::ImageRegionSplitterDirection::Pointer splitter = itk::ImageRegionSplitterDirection::New();
    splitter->SetDirection(2); // Prevent splitting along z axis. As a result, splitting will be performed along y axis
    streamerBP->SetRegionSplitter(splitter);

    typedef itk::ImageFileWriter< OutputImageType > WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName("recons.mha");
    writer->SetInput(streamerBP->GetOutput());
    CHECK_ERROR(writer->Update())

#ifdef WINDOWS
    QueryPerformanceCounter(&end);
    cout << "Reconstruction ended in "
        << (end.QuadPart - start.QuadPart)
        << " units, i.e. " << (end.QuadPart - start.QuadPart)*1.0 / f.QuadPart << " sec" << std::endl;
#else
    end=clock();
    cout << "Reconstruction ended in "
        << end-start
        << " clocks, i.e. " << (end - start)*1.0 / CLOCKS_PER_SEC << " sec" << std::endl;
#endif
}
