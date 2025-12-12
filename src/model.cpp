#include<iostream>
#include"model.hpp"
#include"utils.hpp"
#include"logger.hpp"


Model::Model(int inputnodes, int hiddennodes, int outputnodes, float learningrate)
    : inodes(inputnodes), hnodes(hiddennodes), onodes(outputnodes), lr(learningrate) {

        double stddev1 = std::pow(inodes, -0.5);
        double stddev2 = std::pow(hnodes, -0.5);

        auto wih_opt = initialWeight(hnodes, inodes, 0.0, stddev1);
        if (wih_opt.has_value()) {
            wih = wih_opt.value();
        } else {
            throw std::runtime_error("Failed to initialize wih");
        }
        printMatrixDimensions("wih", wih);

        
        auto who_opt = initialWeight(onodes, hnodes, 0.0, stddev2);
        if (who_opt.has_value()) {
            who = who_opt.value();
        } else {
            throw std::runtime_error("Failed to initialize who");
        }
        printMatrixDimensions("who", who);
    }

void Model::train(std::vector<double> inputs_list, std::vector<double> targets_list) {
    Eigen::MatrixXd inputs = Eigen::Map<Eigen::MatrixXd>(inputs_list.data(), inodes, 1);
    Eigen::MatrixXd targets = Eigen::Map<Eigen::MatrixXd>(targets_list.data(), onodes, 1);
    Eigen::MatrixXd hidden_inputs = wih * inputs;
    Eigen::MatrixXd hidden_outputs = sigmoid(hidden_inputs);

    Eigen::MatrixXd final_inputs = who * hidden_outputs;
    Eigen::MatrixXd final_outputs = sigmoid(final_inputs);


    Eigen::MatrixXd output_errors = targets - final_outputs;
    double sum_error = output_errors.sum();

    LOG("output_errors = %f", sum_error);

    Eigen::MatrixXd hidden_errors = who.transpose() * output_errors;

    who += lr * (output_errors.array() * final_outputs.array() * (1.0 - final_outputs.array())).matrix() * hidden_outputs.transpose();

    wih += lr * (hidden_errors.array() * hidden_outputs.array() * (1.0 - hidden_outputs.array())).matrix() * inputs.transpose();


}

Eigen::MatrixXd Model::forward(std::vector<double> inputs_list) {
    Eigen::MatrixXd inputs = Eigen::Map<Eigen::MatrixXd>(inputs_list.data(), inodes, 1);
    printMatrixDimensions("inputs", inputs);
    Eigen::MatrixXd hidden_inputs = wih * inputs;
    printMatrixDimensions("hidden_inputs", hidden_inputs);

    Eigen::MatrixXd hidden_outputs = sigmoid(hidden_inputs);

    Eigen::MatrixXd final_inputs = who * hidden_outputs;
    Eigen::MatrixXd final_outputs = sigmoid(final_inputs);
    return final_outputs;
}

Eigen::MatrixXd Model::sigmoid(const Eigen::MatrixXd& x) {
    return 1.0 / (1.0 + (-x.array()).exp());
}

Eigen::MatrixXd Model::de_sigmoid(const Eigen::MatrixXd& x) {
    return x.array() * (1.0 - x.array());
}
