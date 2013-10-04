#include <confdefs.h>

#include <vector>
#include <set>
#include <list>

#ifdef DOUBLEP
#define REAL double
#else
#define REAL float
#endif

typedef std::set<std::vector<REAL> > vecset;
typedef std::list<vecset> veclist;
veclist L;

extern "C"
{
  void F77_FUNC_(vec_create_set,VEC_CREATE_SET)(int* idx);
  void F77_FUNC_(vec_is_present,VEC_IS_PRESENT)(int* idx, REAL* arr, int* size, int* success);
  void F77_FUNC_(vec_clear_set,VEC_CLEAR_SET)(int* idx);
  void F77_FUNC_(vec_destroy_set,VEC_DESTROY_SET)(unsigned int* idx);
}

void F77_FUNC_(vec_create_set,VEC_CREATE_SET)(int* idx)
{
  vecset S;
  L.push_back(S);
  *idx = L.size();
}

void F77_FUNC_(vec_is_present,VEC_IS_PRESENT)(int* idx, REAL* arr, int* size, int* success)
{
  std::vector<REAL> v(*size);
  std::pair<vecset::iterator,bool> stat;
  int i;
  vecset S;

  veclist::iterator j;
  int k;
  j = L.begin();
  for (k = *idx; k > 0; k--)
    j++;
  S = *j;

  for (i = 0; i < *size; i++)
    v[i] = arr[i];

  stat = S.insert(v);
  *success = stat.second;
}

void F77_FUNC_(vec_clear_set,VEC_CLEAR_SET)(int* idx)
{
  vecset S;
  veclist::iterator j;
  int k;
  j = L.begin();
  for (k = *idx; k > 0; k--)
    j++;
  S = *j;
  S.clear();
}

void F77_FUNC_(vec_destroy_set,VEC_DESTROY_SET)(unsigned int* idx)
{
  vecset S;
  veclist::iterator j;
  int k;
  j = L.begin();
  for (k = *idx; k > 0; k--)
    j++;
  S = *j;
  S.clear();
  if (L.size() == *idx)
  {
    L.pop_back();
  }
}

