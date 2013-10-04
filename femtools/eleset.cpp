#include <confdefs.h>

#include <set>

#ifdef DOUBLEP
#define REAL double
#else
#define REAL float
#endif

typedef std::set<int> eleset;
eleset E;

extern "C"
{
  void F77_FUNC_(ele_add_to_set,ELE_ADD_TO_SET)(int* element);
  void F77_FUNC_(ele_get_size,ELE_GET_SIZE)(int* size);
  void F77_FUNC_(ele_fetch_list,ELE_FETCH_LIST)(int* arr);
  void F77_FUNC_(ele_get_ele,ELE_GET_ELE)(int* i, int* ele);
}

void F77_FUNC_(ele_add_to_set,ELE_ADD_TO_SET)(int* element)
{
  E.insert(*element);
}

void F77_FUNC_(ele_get_size,ELE_GET_SIZE)(int* size)
{
  *size = E.size();
}

void F77_FUNC_(ele_fetch_list,ELE_FETCH_LIST)(int* arr)
{
  int pos;

  pos = 0;
  for (eleset::const_iterator i = E.begin(); i != E.end(); i++)
  {
    arr[pos++] = *i;
  }

  E.clear();
}

void F77_FUNC_(ele_get_ele,ELE_GET_ELE)(int* i, int* ele)
{
  int pos;

  pos = 0;
  for (eleset::const_iterator j = E.begin(); j != E.end(); j++)
  {
    pos = pos + 1;
    if (pos == *i)
    {
      *ele = *j;
      return;
    }
  }
}

