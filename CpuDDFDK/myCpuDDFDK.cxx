//#include "rtkConfiguration.h"
#include <itkImage.h>
#include <rtkProjectionsReader.h>
#include <rtkThreeDCircularProjectionGeometryXMLFile.h>
#include <rtkConstantImageSource.h>
#include <rtkFDKWeightProjectionFilter.h>
#include <rtkFFTRampImageFilter.h>
//#include <rtkBackProjectionImageFilter.h>
//#include <rtkFDKBackProjectionImageFilter.h>
//#include "myBackProjectionImageFilter.h"
//#include "myDDBackProjectionImageFilter.h"
#include <itkStreamingImageFilter.h>
#include <itkImageRegionSplitterDirection.h>
#include "ddfdk.h"
#include <iostream>
using namespace std;

int main(int argc, char*argv[]) {
    //Typedef
    typedef float OutputPixelType;
    const unsigned int Dimension = 3;
    typedef itk::Image< OutputPixelType, Dimension >     OutputImageType;

    //Projection Reader
    typedef rtk::ProjectionsReader< OutputImageType > ReaderType;
    ReaderType::Pointer reader = ReaderType::New();

    std::vector<std::string> filenames;
    filenames.push_back("projections.mha");
    reader->SetFileNames(filenames);
    try { reader->UpdateOutputInformation(); }
    catch (itk::ExceptionObject &err) {
        cerr << "异常！" << endl;
        cerr << err << endl;
        return 1;
    }

    //Geometry
    rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader;
    geometryReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
    geometryReader->SetFilename("geometry.xml");
    try { geometryReader->GenerateOutputInformation(); }
    catch (itk::ExceptionObject &err) {
        cerr << "异常！" << endl;
        cerr << err << endl;
        return 1;
    }
    

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
    try { constantImageSource->UpdateOutputInformation(); }
    catch (itk::ExceptionObject &err) {
        cerr << "异常！" << endl;
        cerr << err << endl;
        return 1;
    }

    //********* FDK 重建 **********//
    typedef itk::ExtractImageFilter<OutputImageType, OutputImageType>                   ExtractFilterType;
    typename ExtractFilterType::Pointer                                                 ExtractFilter;
    ExtractFilter = ExtractFilterType::New();
    typedef rtk::FDKWeightProjectionFilter<OutputImageType, OutputImageType>            WeightFilterType;
    typename WeightFilterType::Pointer                                                  WeightFilter;
    WeightFilter = WeightFilterType::New();
    typedef rtk::FFTRampImageFilter<OutputImageType, OutputImageType, double>           RampFilterType;
    typename RampFilterType::Pointer                                                    RampFilter;
    RampFilter = RampFilterType::New();
    typedef DDBackProjectionImageFilter<OutputImageType, OutputImageType>               BackProjectionFilterType;
    typename BackProjectionFilterType::Pointer                                          BackProjectionFilter;
    BackProjectionFilter = BackProjectionFilterType::New();

    //加权
    WeightFilter->SetInput(reader->GetOutput());
    WeightFilter->InPlaceOn();
    WeightFilter->SetGeometry(geometryReader->GetOutputObject());

    //滤波
    RampFilter->SetInput(WeightFilter->GetOutput());
    RampFilter->SetTruncationCorrection(0.0);
    RampFilter->SetHannCutFrequency(0.0);
    RampFilter->SetHannCutFrequencyY(0.0);

    //反投影
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

    try { writer->Update(); }
    catch (itk::ExceptionObject &err) {
        cerr << "异常！" << endl;
        cerr << err << endl;
        return 1;
    }
}