/*
* This file is part of htmap.
*
* Copyright (C) 2018 Emilio Garcia-Fidalgo <emilio.garcia@uib.es> (University of the Balearic Islands)
*
* htmap is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* htmap is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with htmap. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef _GLOBALDESCRIPTOR_H_
#define _GLOBALDESCRIPTOR_H_

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "htmap/util/Util.h"
#include "ldb.h"

namespace htmap
{

enum GlobalDescriptorType
{
    GDESCRIPTOR_WISIFT,
    GDESCRIPTOR_WISURF,
    GDESCRIPTOR_BRIEFGIST,
    GDESCRIPTOR_WILDB,
    GDESCRIPTOR_PHOG,
    GDESCRIPTOR_RESNET,
};

// ---
// General parameters for global descriptors
// ---
struct GlobalDescriptorParams
{
    GlobalDescriptorParams()
	{}
};

// ---
// Abstract global descriptor class.
// ---
class GlobalDescriptor
{
	public:
        GlobalDescriptor(const GlobalDescriptorType& type, const int nbytes) :
			_type(type),
			_desc_size(nbytes)
        {}
        virtual ~GlobalDescriptor()
        {}

        inline int getDescSize() { return _desc_size; }
        inline GlobalDescriptorType getType() { return _type; }
        static double dist(const cv::Mat& a, const cv::Mat& b, GlobalDescriptor* desc);
        static double dist(const cv::Mat& a, const cv::Mat& b, const cv::Mat& icovar);

        static GlobalDescriptor* create(const std::string& name, const GlobalDescriptorParams& params);

        virtual void parseParameters(const GlobalDescriptorParams& params) = 0;
        virtual void describe(const cv::Mat& image, cv::Mat& desc) = 0;

	protected:
        GlobalDescriptorType _type;
		int _desc_size;
};

// ---
// WI-SIFT descriptor class.
// ---
class WISIFTDescriptor : public GlobalDescriptor
{
	public:
        WISIFTDescriptor(const GlobalDescriptorParams& params) :
            GlobalDescriptor(GDESCRIPTOR_WISIFT, 256)
	{
			parseParameters(params);
    }

    void parseParameters(const GlobalDescriptorParams& params)
	{
    }

    void describe(const cv::Mat& image, cv::Mat& desc);
};

// ---
// WI-SURF descriptor class.
// ---
class WISURFDescriptor : public GlobalDescriptor
{
    public:
        WISURFDescriptor(const GlobalDescriptorParams& params) :
            GlobalDescriptor(GDESCRIPTOR_WISURF, 256)
    {
            parseParameters(params);
    }

    void parseParameters(const GlobalDescriptorParams& params)
    {
    }

    void describe(const cv::Mat& image, cv::Mat& desc);
};

// ---
// BRIEF-Gist descriptor class.
// ---
class BRIEFGistDescriptor : public GlobalDescriptor
{
    public:
        BRIEFGistDescriptor(const GlobalDescriptorParams& params) :
            GlobalDescriptor(GDESCRIPTOR_BRIEFGIST, 64)
    {
            parseParameters(params);
    }

    void parseParameters(const GlobalDescriptorParams& params)
    {
    }

    void describe(const cv::Mat& image, cv::Mat& desc);
};

// ---
// WI-LDB descriptor class.
// ---
class WILDBDescriptor : public GlobalDescriptor
{
    public:
        WILDBDescriptor(const GlobalDescriptorParams& params) :
            GlobalDescriptor(GDESCRIPTOR_WILDB, 64)
    {
            parseParameters(params);
    }

    void parseParameters(const GlobalDescriptorParams& params)
    {
    }

    void describe(const cv::Mat& image, cv::Mat& desc);

    private:
        LdbDescriptorExtractor _ldb;
};

// ---
// PHOG descriptor class.
// ---
class PHOGDescriptor : public GlobalDescriptor
{
    public:
        PHOGDescriptor(const GlobalDescriptorParams& params) :
            GlobalDescriptor(GDESCRIPTOR_PHOG, 420) // 20 Bins (18 deg per bin), and L = 3
    {
            parseParameters(params);
    }

    void parseParameters(const GlobalDescriptorParams& params)
    {
    }

    void describe(const cv::Mat& image, cv::Mat& desc);

private:
    void getHistogram(const cv::Mat& edges, const cv::Mat& ors, const cv::Mat& mag, int startX, int startY, int width, int height, cv::Mat& hist);
};

class RESNETDescriptor : public GlobalDescriptor
{
    public:
        void readImg(std::string, cv::Mat&);
        RESNETDescriptor(const GlobalDescriptorParams& params) :
            GlobalDescriptor(GDESCRIPTOR_RESNET, 2048), // 2048 embedding size
            device(torch::kCPU), cpu(torch::kCPU)
    {
            parseParameters(params);
            setupDevice();
            loadModel();
    }

    void parseParameters(const GlobalDescriptorParams& params)
    {
    }

    void describe(const cv::Mat& image, cv::Mat& desc);

    private:
        torch::Device cpu;
        torch::Device device;
        torch::jit::script::Module model;
        void loadModel();
        void setupDevice();
        void resizeNormalizeImg(const cv::Mat&, cv::Mat&);
        void getInputTensor(const cv::Mat&, torch::Tensor&);
};

}

#endif /* GLOBALDESCRIPTOR_H_ */
