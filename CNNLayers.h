#pragma once
#include <omp.h>
#include <vector>
#include <iostream>
#include "NetBase.h"

struct CNN_matrix {
  std::vector<float> matx;
  size_t width = 0;
};

struct CNN_cell {
  CNN_matrix core;
  CNN_matrix matrix;
};

typedef std::vector<CNN_cell> CNN_layer;

enum class Pass_type {
  Valid,
  Same,
  Full
};

struct CNN_init_struct {
  size_t outputs_count = 2;
  size_t outputs_row_width = 10;
  size_t layers_count = 2;
  size_t input_row_width = 10;
  size_t input_length = 20;
  size_t output_length = 20;
  Pass_type p_type = Pass_type::Same;
};

class CNNLayers
{
protected:
  CNN_matrix input;
  std::vector<CNN_matrix> output;
  std::vector<CNN_layer> layers;
  CNN_init_struct dimensions;
  void PassFunc(CNN_matrix &source, CNN_matrix &core, CNN_matrix &res, activation_func act_func);
  void ValidPass(CNN_matrix &source, CNN_matrix &core, CNN_matrix &res, activation_func act_func);
public:
  CNNLayers();
  void InitCNN(CNN_init_struct &t);
  void PassForward(CNN_matrix &inp, std::vector<CNN_matrix> &out, activation_func act_func);
  void Test();
  ~CNNLayers();
};

