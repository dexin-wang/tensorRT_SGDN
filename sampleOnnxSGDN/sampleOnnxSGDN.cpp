//! sampleOnnxSGDN.cpp
//  使用sgdn.pnnx进行推理
//! It can be run with the following command line:
//! Command: ./sample_onnx_sgdn [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

const std::string gSampleName = "TensorRT.sample_onnx_sgdn";

//! \brief  The SampleOnnxMNIST class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!
class SampleOnnxMNIST
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleOnnxMNIST(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();
    void readBinImg(std::string& filename, std::vector<uint8_t>& data);
    void readTxtImg(std::string& filename, std::vector<int>& data, int nlines);

private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleOnnxMNIST::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);     
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    // 使用onnx模型创建tensorRT模型
    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    // 修改输入输出维度
    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 4);

    assert(network->getNbOutputs() == 3);
    mOutputDims = network->getOutput(0)->getDimensions();
    assert(mOutputDims.nbDims == 4);

    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleOnnxMNIST::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(
        locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(16_MiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleOnnxMNIST::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }
    cout << "========================================== SampleUniquePtr done ==========================================" << endl;

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        cout << "========================================== processInput 5 ==========================================" << endl;
        return false;
    }
    cout << "========================================== read img done ==========================================" << endl;

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }
    cout << "========================================== executeV2 done ==========================================" << endl;
    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }
    cout << "========================================== verify Output done ==========================================" << endl;

    return true;
}


// 从txt文件读取图像数据  按照BGR的顺序
void SampleOnnxMNIST::readTxtImg(std::string& filename, std::vector<int>& data, int nlines)
{
    std::ifstream infile(filename, ios::in);

    int id = 0;
    string line;

    const char pattern = ' ';
    for(int i = 0; i < nlines; i++)
    {
        getline(infile, line);
        stringstream input(line);   //读取line到字符串流中
        string temp;
        while(getline(input, temp, pattern))
        {
            data[id++] = std::stoi(temp);
        }
    }

    infile.close();
}


// 从二进制文件读取图像数据  按照BGR的顺序
void SampleOnnxMNIST::readBinImg(std::string& filename, std::vector<uint8_t>& data)
{
    std::ifstream infile(filename, ios::in|ios::binary);
    uint8_t val;
    int i = 0;
    while(infile.read((char*)&val, sizeof(uint8_t)))
    {
        cout << i << " ";
        data[i++] = val;
    }
    infile.close();
}


//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleOnnxMNIST::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputC = mInputDims.d[1]; // 3
    const int inputH = mInputDims.d[2]; // 320
    const int inputW = mInputDims.d[3]; // 320

    // Read a random digit file
    srand(unsigned(time(nullptr)));
    std::vector<int> fileData(inputC * inputH * inputW);

    // 从二进制文件读取图像数据  按照BGR的顺序
    std::string imgpath = "/home/wangdx/tensorRT/TensorRT-7.0.0.11/samples/sampleOnnxSGDN/data/img.txt";
    readTxtImg(imgpath, fileData, inputC*inputH);

    // 将图像数据填充至缓冲区
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    const float pixelMean[3] = {102.9801f, 115.9465f, 122.7717f}; // Also in BGR order
    int volChl = inputH * inputW;
    // cout << "inputC = " << inputC << "inputH = " << inputH << " inputW = " << inputW << " volChl = " << volChl << endl;
    for (int c = 0; c < inputC; c++)
    {
        // The color image to input should be in BGR order
        for (int j = 0; j < volChl; j++)
        {
            // cout << c * volChl + j << " ";
            hostDataBuffer[c * volChl + j] = (float(fileData[c * volChl + j]) - pixelMean[c]) / 255.0;
        }
            
    }
    

    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool SampleOnnxMNIST::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputH = mOutputDims.d[2];   // 320
    const int outputW = mOutputDims.d[3];   // 320
    float* output_able = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));      // (1,   1, 320, 320)
    float* output_angle = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[1]));     // (1, 120, 320, 320)
    float* output_width = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[2]));     // (1,   1, 320, 320)
    float confidence = 0.0f;
    int idx = 0;

    gLogInfo << "Output:" << std::endl;

    // 计算sigmoid
    for (int i = 0; i < outputH*outputW; i++)
    {
        output_able[i] = 1.0 / (1.0 + exp(-1 * output_able[i]));
        confidence = std::max(confidence, output_able[i]);
        if (confidence == output_able[i])
        {
            idx = i;
        }
    }

    int row = idx / outputW;
    int col = idx % outputW;

    gLogInfo << "(row, col) = " << row << ", " << col << endl;
    cout << "confidence = " << confidence << endl;

    return confidence > 0.5f;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("/home/wangdx/tensorRT/TensorRT-7.0.0.11/samples/sampleOnnxSGDN/data/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "sgdn.onnx";
    params.inputTensorNames.push_back("input");
    params.batchSize = 1;
    params.outputTensorNames.push_back("output_able");
    params.outputTensorNames.push_back("output_angle");
    params.outputTensorNames.push_back("output_width");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_onnx_sgdn [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help) 
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    SampleOnnxMNIST sample(initializeSampleParams(args));

    gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;

    if (!sample.build())
    {
        return gLogger.reportFail(sampleTest);
    }
    cout << "========================================== build done ==========================================" << endl;
    if (!sample.infer())
    {
        return gLogger.reportFail(sampleTest);
    }
    cout << "========================================== infer done ==========================================" << endl;

    return gLogger.reportPass(sampleTest);
}
