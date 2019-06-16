#include "CNNLayers.h"


void CNNLayers::PassFunc(CNN_matrix &source, CNN_matrix &core, CNN_matrix &res, activation_func act_func)
{

}

void CNNLayers::ValidPass(CNN_matrix & source, CNN_matrix & core, CNN_matrix & res, activation_func act_func)
{
  size_t core_offset = (core.width - 1) / 2;
  size_t source_w = 0;
  size_t source_h = 0;
  size_t source_s = 0;
  size_t source_off = 0;
  for (size_t g = 0; g < res.matx.size(); ++g)
  {
    size_t core_h = 0;
    size_t core_w = 0;
    source_off = (source_off + core_offset > res.width) ? 0 : (g != 0 ? source_off + 1 : 0);
    source_w = source_off;
    source_s = (g % res.width == 0 && g != 0) ? source_s + 1 : source_s;
    source_h = source_s;
    for (size_t i = 0; i < core.matx.size(); ++i)
    {
      res.matx[g] += source.matx[(source_h * source.width) + source_w] * core.matx[(core_h * core.width) + core_w];
      core_w++;
      source_w++;
      if (core_w == core.width)
      {
        core_h++;
        source_h++;
        source_w = source_off;
        core_w = 0;
      }
    }
    if (act_func != nullptr)
      res.matx[g] = act_func(res.matx[g], 0.05);
  }
}

void CNNLayers::SamePass(CNN_matrix & source, CNN_matrix & core, CNN_matrix & res, activation_func act_func)
{
  size_t offset = (core.width - 1) / 2;
  CNN_matrix tmp;
  tmp.width = source.width + (offset * 2);
  tmp.matx.resize(source.matx.size() + ((tmp.width * offset) * 2) + ((source.matx.size() / source.width) * offset * 2));
  size_t j = 0;
  size_t k = 0;
  for (size_t i = 0; i < tmp.matx.size(); ++i)
  {
    if (j >= tmp.width) 
    { 
      j = 0;
    }
    if ((i < tmp.width * offset) || (i > tmp.matx.size() - tmp.width * offset))
    {
      tmp.matx[i] = 0;
      continue;
    }
    if ((j < offset) || (j >= tmp.width - offset))
    {
      tmp.matx[i] = 0;
      j++;
      continue;
    }

    tmp.matx[i] = source.matx[k];
    k++;
    j++;
  }
  j = 0;
  for (size_t i = 0; i < tmp.matx.size(); ++i)
  {
    j++;
    std::cout << tmp.matx[i] << " ";
    if (j == tmp.width) {
      j = 0;
      std::cout << std::endl;
    }
  }

  ValidPass(tmp, core, res, act_func);
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
  res.width = 10;
  res.matx.resize(100);
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
  
  SamePass(t, core, res, nullptr);
  cout << endl << endl;
  j = 0;
  for (size_t i = 0; i < res.matx.size(); ++i)
  {
    
    if (j == res.width)
    {
      j = 0;
      cout << endl;
    }
    j++;
    cout << res.matx[i] << " ";
  }
  cout << endl << endl;

}

CNNLayers::~CNNLayers()
{
}
