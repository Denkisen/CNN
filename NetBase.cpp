#include "NetBase.h"

void NetBase::Warning(std::string text)
{
  std::cout << "Warning: " << text << std::endl;
}

void NetBase::SetDefaults()
{
  if (state == NState::Running)
  {
    Warning("Net Already in Running state");
    return;
  }
  if (state == NState::NotConfigured)
  {
    Warning("Net is in NotConfigured state");
    return;
  }

  std::srand((unsigned) time(nullptr));

  for (auto i = 0; i < net.size(); ++i)
  {
    for (auto j = 0; j < net[i].size(); ++j)
    {
      std::get<0>(net[i][j]) = 0;
      std::get<2>(net[i][j]) = 0;
      for (auto l = 0; l < std::get<1>(net[i][j]).size(); ++l)
      {
        std::get<1>(net[i][j])[l] = (0.5 - (double)(rand()) / RAND_MAX) * 0.5;
      }
    }
  }

  state = NState::Ready; 
}

NetBase::NetBase()
{
  omp_set_num_threads(6);
}

void NetBase::SetLayarsCount(size_t layers)
{
  if (state == NState::Running)
  {
    Warning("Net Already in Running state");
    return;
  }

  if (layers < MIN_LAYERS_COUNT)
    std::exception("too small layers count");

  state = NState::NotConfigured;
  net.clear();
  net.resize(layers);
}

void NetBase::SetLayarSize(size_t index, size_t neurons)
{
  if (state == NState::Running)
  {
    Warning("Net Already in Running state");
    return;
  }
  
  if (index >= net.size())
    std::exception("out off layers range");

  if (neurons < MIN_NEURONS_COUNT)
    std::exception("too small neurons count");

  state = NState::NotConfigured;
  net[index].clear();
  net[index].resize(neurons);
  if (index != 0)
  {
    for (auto i = 0; i < net[index - 1].size(); ++i)
    {
      std::get<1>(net[index - 1][i]).resize(neurons);
    }
  }

}

void NetBase::MakeNet()
{
  if (state == NState::Running)
  {
    Warning("Net Already in Running state");
    return;
  }
  auto count = net.size();
  bool good = true;
  for (auto i = 0; i < count; ++i)
  {
    if (net[i].size() < MIN_NEURONS_COUNT)
    {
      good = false;
      break;
    }
  }
  if(!good)
    std::exception("not enough neurons count");

  state = NState::Configured;
  SetDefaults();
}

void NetBase::MakeNet(std::vector<std::vector<std::vector<long double>>> &weights)
{
  if (state == NState::Running)
  {
    Warning("Net Already in Running state");
    return;
  }

  if (weights.size() < MIN_LAYERS_COUNT)
    std::exception("too small layers count");

  bool good = true;
  for (auto i = 0; i < weights.size(); ++i)
  {
    if (weights[i].size() < MIN_NEURONS_COUNT)
    {
      good = false;
      break;
    }
  }

  SetLayarsCount(weights.size());

  for (auto i = 0; i < weights.size(); ++i)
  {
    SetLayarSize(i, weights[i].size());
  }

  for (auto i = 0; i < weights.size(); ++i)
  {
    for (auto j = 0; j < weights[i].size(); ++j)
    {
      for (auto l = 0; l < weights[i][j].size(); ++l)
        std::get<1>(net[i][j])[l] = weights[i][j][l];
    }
  }

  state = NState::Ready;
}

void NetBase::PassForward(std::vector<double> &data, std::vector<double> &out)
{
  if (state == NState::Running)
  {
    Warning("Net Already in Running state");
    return;
  }
  if (act_f == nullptr)
    std::exception("no activation function");
  if (state != NState::Ready)
    std::exception("net not ready");
  if (data.size() != net[0].size())
    std::exception("input != data");
  if (out.size() != net[net.size() - 1].size())
    std::exception("output != out");

  state = NState::Running;

#pragma omp parallel for if (data.size() > 1000)
  for (auto i = 0; i < data.size(); ++i)
  {
    std::get<0>(net[0][i]) = data[i];
  }

  for (auto i = 0; i < net.size(); ++i)
  {
#pragma omp parallel for if (net[i].size() > 1000)
    for (auto j = 0; j < net[i].size(); ++j)
    {
      if (i == 0)
      {
        std::get<2>(net[i][j]) = act_f(std::get<0>(net[i][j]), alpha);
      }
      else
      {
        std::get<0>(net[i][j]) = 0;
        for (auto k = 0; k < net[i - 1].size(); ++k)
        {
          std::get<0>(net[i][j]) += std::get<1>(net[i - 1][k])[j] * std::get<2>(net[i - 1][k]);
        }
        std::get<2>(net[i][j]) = act_f(std::get<0>(net[i][j]), alpha);
      }
    }
  }

  auto last = net.size() - 1;
#pragma omp parallel for if (out.size() > 1000)
  for (auto i = 0; i < out.size(); ++i)
  {
    out[i] = std::get<2>(net[last][i]);
  }

  state = NState::Ready;
}

void NetBase::SetActivation(activation_func func, long double alpha)
{
  act_f = func;
  this->alpha = alpha;
}


NetBase::~NetBase()
{
}
