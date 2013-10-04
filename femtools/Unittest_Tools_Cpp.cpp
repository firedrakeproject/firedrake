#include "Unittest_Tools_Cpp.h"

using namespace std;

//* Used by unit tests to report the output of a test case
void report_test(const string& title, const bool& fail, const bool& warn, const string& msg){
  if(fail){
    cout << "Fail: " << title << "; error: " << msg << "\n";
  }else if(warn){
    cout << "Warn: " << title << "; error: " << msg << "\n";
  }else{
    cout << "Pass: " << title << "\n";
  }

  return;
}

void cget_nan(double *nan)
{
  *nan = std::numeric_limits<double>::quiet_NaN();
}
