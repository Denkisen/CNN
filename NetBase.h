#pragma once
#include <omp.h>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <random>
#include <time.h>

#define MIN_LAYERS_COUNT 3
#define MIN_NEURONS_COUNT 3

typedef void func (std::tuple<long double, long double, long double> &rec);
typedef long double activation_func (long double x, long double alpha);

enum class NState {
  Running,
  Ready,
  Configured,
  NotConfigured
};

class NetBase
{
protected:
  NState state = NState::NotConfigured;
  std::vector<std::vector<std::tuple<long double, std::vector<long double>, long double>>> net; // weights[layer][neuron](input ,weights[], result)
  void Warning(std::string text);
private:
  void SetDefaults();
  activation_func *act_f = nullptr;
  long double alpha = 0.01; // function's angle
public:
  NetBase();
  void SetLayarsCount(size_t layers);
  void SetLayarSize(size_t index, size_t neurons);
  void MakeNet();
  void MakeNet(std::vector<std::vector<std::vector<long double>>> &weights);
  void PassForward(std::vector<double> &data, std::vector<double> &out);
  void SetActivation(activation_func func, long double alpha);
  auto GetNet() { return &net; }
  auto GetState() { return state; }
  ~NetBase();
};

