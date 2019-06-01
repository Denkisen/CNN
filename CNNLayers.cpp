#include "CNNLayers.h"


void CNNLayers::PassFunc(CNN_matrix &source, CNN_matrix &core, CNN_matrix &res, activation_func act_func)
{
 
}

void CNNLayers::ValidPass(CNN_matrix & source, CNN_matrix & core, CNN_matrix & res, activation_func act_func)
{
  size_t s_i = 0;
  size_t s_j = 0;
  size_t w_j = 0;
  size_t w_i = 0;

  for (size_t g = 0; g < res.matx.size(); ++g)
  {
    s_i = w_j;
    s_j = w_i;
    for (size_t i = 0; i < core.matx.size(); ++i)
    {
      if (i != 0 && i % core.width == 0)
      {
        s_j++;
        s_i = w_j;
      }
      res.matx[g] += source.matx[(s_j * source.width) + s_i] * core.matx[i];
      s_i++;
    }
    w_j++;
    if (w_j > core.width)
    {
      w_j = 0;
      w_i++;
    }
  }
}

CNNLayers::CNNLayers()
{
}

void CNNLayers::PassForward(CNN_matrix & inp, std::vector<CNN_matrix>& out, activation_func act_func)
{
  if (inp.matx.size() == 0)
    std::exception("empty inp matrix");
  if (out.size() == 0)
    std::exception("empty out matrix");
  if (act_func == nullptr)
    std::exception("empty act_func matrix");
  if (inp.matx.size() != dimensions.input_length)
    std::exception("wrong dimensions: input_length");
  if (inp.width != dimensions.input_row_width)
    std::exception("wrong dimensions: input_row_width");
  if (out.size() != dimensions.outputs_count)
    std::exception("wrong dimensions: outputs_count");
 

}

void CNNLayers::Test()
{
  using namespace std;
  CNN_matrix t;
  t.matx.resize(100);
  t.width = 10;
  CNN_matrix res;
  res.width = 6;
  res.matx.resize(36);
  CNN_matrix core;
  core.width = 5;
  core.matx.resize(25);
 
  for (size_t i = 0; i < core.matx.size(); ++i)
  {
    core.matx[i] = 0;
  }
  core.matx[12] = 1;
  size_t j = 0;
  for (size_t i = 0; i < t.matx.size(); ++i)
  {
    if (i != 0 && i % t.width == 0)
    {
      j++;
      cout << endl;
    }
    t.matx[i] = i;
    cout << t.matx[i] << " ";
  }
  cout << endl << endl;
  
  ValidPass(t, core, res, nullptr);

  j = 0;
  for (size_t i = 0; i < res.matx.size(); ++i)
  {
    if (i != 0 && i % res.width == 0)
    {
      j++;
      cout << endl;
    }
    cout << res.matx[i] << " ";
  }
  cout << endl << endl;

}

CNNLayers::~CNNLayers()
{
}
