#include "GaussNewtonOptimizer.h"
#include <iostream>

GaussNewtonOptimizer::GaussNewtonOptimizer(std::vector<double> xValues, std::vector<double> yValues, int maxIteration, double invStdDeviation)
    : xData(xValues), yData(yValues), iteration(maxIteration), invSigma(invStdDeviation), n(xValues.size()) {}

void GaussNewtonOptimizer::setInitialParams(std::vector<double> initialParams) {
    parameters = Eigen::Map<Eigen::VectorXd>(initialParams.data(), initialParams.size());
    numParameters = initialParams.size();
}

Eigen::VectorXd GaussNewtonOptimizer::getEstimatedParams() const {
    return parameters;
}

void GaussNewtonOptimizer::fit() {
    double cost = 0, lastCost = 0;
    auto t1 = std::chrono::steady_clock::now();

    int pSize = parameters.size();
    Eigen::VectorXd dx(pSize);

    for (int iter = 0; iter < iteration; ++iter) {
        Eigen::MatrixXd H(pSize, pSize);
        H.setZero();
        Eigen::VectorXd b(pSize);
        b.setZero();
        cost = 0;

        for (int i = 0; i < n; ++i) {
            double xi = xData[i], yi = yData[i];
            double error = GaussNewtonOptimizer::computeError(xi, yi, parameters);
            Eigen::VectorXd J = GaussNewtonOptimizer::computeJacobian(xi, parameters);

            H += invSigma * invSigma * J * J.transpose();
            b += -invSigma * invSigma * error * J;

            cost += error * error;
        }

        dx = H.ldlt().solve(b);
        if (dx.array().isNaN().any()) {
            std::cout << "Result is nan" << std::endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            std::cout << "Cost: " << cost << " >= last cost: " << lastCost << ", break" << std::endl;
            break;
        }

        parameters += dx;
        lastCost = cost;

        std::cout << "Total cost: " << cost << ", update: " << dx.transpose() << ", estimated params: " << parameters.transpose() << std::endl;
    }

    auto t2 = std::chrono::steady_clock::now();
    std::cout << "Optimization finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;
}


// Calculate the error for a given data point
double GaussNewtonOptimizer::computeError(double xi, double yi, const Eigen::VectorXd& params) {
    double prediction = 0;
    for (int j = 0; j < params.size(); ++j) {
        prediction += params[j] * std::pow(xi, params.size() - j - 1);
    }
    return yi - exp(prediction);
}

// Calculate the Jacobian vector for a given data point
Eigen::VectorXd GaussNewtonOptimizer::computeJacobian(double xi, const Eigen::VectorXd& params) {
    Eigen::VectorXd J(params.size());
    double prediction = 0;
    for (int j = 0; j < params.size(); ++j) {
        prediction += params[j] * std::pow(xi, params.size() - j - 1);
    }
    for (int j = 0; j < params.size(); ++j) {
        J[j] = -std::pow(xi, params.size() - j - 1) * exp(prediction);
    }
    return J;
}