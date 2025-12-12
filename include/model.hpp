#ifndef __MODEL__HPP__
#define __MODEL__HPP__

#include <Eigen/Dense>
#include <iostream>

class Model{
public:
 Model(int inputnodes, int hiddennodes, int outputnodes, float learningrate);
 void train(std::vector<double> inputs_list, std::vector<double> targets_list);
 Eigen::MatrixXd forward(std::vector<double> inputs_list);
 Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x);
 Eigen::MatrixXd de_sigmoid(const Eigen::MatrixXd& x);

private:
    int inodes;
    int hnodes;
    int onodes;
    float lr;
    Eigen::MatrixXd wih; 
    Eigen::MatrixXd who;
};


#endif