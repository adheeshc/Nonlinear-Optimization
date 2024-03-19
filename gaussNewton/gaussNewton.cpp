#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include "GaussNewtonOptimizer.h"

int main(int argc, char** argv) {

    
    double ar = 1.0, br = 2.0, cr = 3.0, dr = 2.0;    // ground truth values
    double ae = 2.0, be = -1.0, ce = 5.0, de = 2.0;   // initial estimate
    int n = 100;                            // number of data points
    double wSigma = 1.0;                    // std dev of data points
    cv::RNG rng;                            // Random number generator
    int maxIterations = 100;                // max iterations to run optimization

    //DATA 1
    // std::vector<double> xData, yData;
    // for (int i = 0; i < n; i++) {
    //     double x = i / 100.0;
    //     xData.emplace_back(x);
    //     yData.emplace_back(exp(ar * x * x + br * x + cr) + rng.gaussian(wSigma * wSigma));
    // }
    // std::vector<double> initialParams{ 2.0, -1.0, 5.0 };

    //DATA 2
    std::vector<double> xData, yData;
    for (int i = 0; i < n; i++) {
        double x = i / 100.0;
        xData.emplace_back(x);
        yData.emplace_back(exp(ar * x * x * x + br * x * x + cr * x + dr) + rng.gaussian(wSigma * wSigma));
    }    
    std::vector<double> initialParams{ 2.0, -1.0, 5.0, 4.0 };

    GaussNewtonOptimizer gaussNewton(xData, yData, maxIterations, wSigma);
    gaussNewton.setInitialParams(initialParams);
    gaussNewton.fit();
    Eigen::VectorXd estimatedParams = gaussNewton.getEstimatedParams();
    
    std::cout << "estimated abc: ";
    for(int i = 0; i<estimatedParams.size();i++){
        std::cout << estimatedParams[i] << ", ";
    }
    std::cout<<std::endl;
    // std::cout << "estimated abc: " << estimatedParams[0] << "," << estimatedParams[1] << "," << estimatedParams[2] << "," << estimatedParams[3] << std::endl;
    return 0;
}