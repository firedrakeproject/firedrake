#include <confdefs.h>

#include <vector>
#include <set>
#include <list>

typedef std::set<std::vector<int> > intvecset;
typedef std::list<intvecset> intveclist;
intveclist IVS;

extern "C"
{
  void F77_FUNC_(intvec_create_set,INTVEC_CREATE_SET)(int* idx);
  void F77_FUNC_(intvec_is_present,INTVEC_IS_PRESENT)(int* idx, int* arr, int* size, int* success);
  void F77_FUNC_(intvec_clear_set,INTVEC_CLEAR_SET)(int* idx);
  void F77_FUNC_(intvec_destroy_set,INTVEC_DESTROY_SET)(unsigned int* idx);
}

void F77_FUNC_(intvec_create_set,INTVEC_CREATE_SET)(int* idx)
{
  intvecset S;
  IVS.push_back(S);
  *idx = IVS.size();
}

void F77_FUNC_(intvec_is_present,INTVEC_IS_PRESENT)(int* idx, int* arr, int* size, int* success)
{
  std::vector<int> v(*size);
  std::pair<intvecset::iterator,bool> stat;
  int i;
  intvecset S;

  intveclist::iterator j;
  int k;
  j = IVS.begin();
  for (k = *idx; k > 0; k--)
    j++;
  S = *j;

  for (i = 0; i < *size; i++)
    v[i] = arr[i];

  stat = S.insert(v);
  *success = stat.second;
}

void F77_FUNC_(intvec_clear_set,INTVEC_CLEAR_SET)(int* idx)
{
  intvecset S;
  intveclist::iterator j;
  int k;
  j = IVS.begin();
  for (k = *idx; k > 0; k--)
    j++;
  S = *j;
  S.clear();
}

void F77_FUNC_(intvec_destroy_set,INTVEC_DESTROY_SET)(unsigned int* idx)
{
  intvecset S;
  intveclist::iterator j;
  int k;
  j = IVS.begin();
  for (k = *idx; k > 0; k--)
    j++;
  S = *j;
  S.clear();
  if (IVS.size() == *idx)
  {
    IVS.pop_back();
  }
}

