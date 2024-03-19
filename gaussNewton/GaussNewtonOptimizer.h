#ifndef GAUSS_NEWTON_OPTIMIZER_H
#define GAUSS_NEWTON_OPTIMIZER_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <chrono>
#include <vector>

class GaussNewtonOptimizer {
private:
    Eigen::VectorXd parameters; // Parameters to estimate
    int numParameters; // Number of parameters
    int iteration; // Maximum number of iterations
    double invSigma; // Inverse of the standard deviation
    std::vector<double> xData, yData; // Input data
    int n; // Number of data points

    double computeError(double xi, double yi, const Eigen::VectorXd& params);
    Eigen::VectorXd computeJacobian(double xi, const Eigen::VectorXd& params);

public:
    GaussNewtonOptimizer(std::vector<double> xValues, std::vector<double> yValues, int maxIteration, double invStdDeviation);
    void setInitialParams(std::vector<double> initialParameters);
    Eigen::VectorXd getEstimatedParams() const;
    void fit();
};

#endif // GAUSS_NEWTON_OPTIMIZER_H
