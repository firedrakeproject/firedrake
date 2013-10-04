// Wild Magic Source Code
// David Eberly
// http://www.geometrictools.com
// Copyright (c) 1998-2008
//
// This library is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 2.1 of the License, or (at
// your option) any later version.  The license is available for reading at
// either of the locations:
//     http://www.gnu.org/copyleft/lgpl.html
//     http://www.geometrictools.com/License/WildMagicLicense.pdf
//
// Version: 4.0.0 (2006/06/28)

#ifndef WM4COMMAND_H
#define WM4COMMAND_H

#include "Wm4System.h"

namespace Wm4
{

class WM4_FOUNDATION_ITEM Command
{
public:
    Command (int iQuantity, char** apcArgument);
    Command (char* acCmdline);
    ~Command ();

    // return value is index of first excess argument
    int ExcessArguments ();

    // Set bounds for numerical arguments.  If bounds are required, they must
    // be set for each argument.
    Command& Min (double dValue);
    Command& Max (double dValue);
    Command& Inf (double dValue);
    Command& Sup (double dValue);

    // The return value of the following methods is the option index within
    // the argument array.

    // Use the boolean methods for options which take no argument, for
    // example in
    //           myprogram -debug -x 10 -y 20 filename
    // the option -debug has no argument.

    int Boolean (char* acName);  // returns existence of option
    int Boolean (char* acName, bool& rbValue);
    int Integer (char* acName, int& riValue);
    int Float (char* acName, float& rfValue);
    int Double (char* acName, double& rdValue);
    int String (char* acName, char*& racValue);
    int Filename (char*& racName);

    // last error reporting
    const char* GetLastError ();

protected:
    // constructor support
    void Initialize ();

    // command line information
    int m_iQuantity;       // number of arguments
    char** m_apcArgument;  // argument list (array of strings)
    char* m_acCmdline;     // argument list (single string)
    bool* m_abUsed;        // keeps track of arguments already processed

    // parameters for bounds checking
    double m_dSmall;   // lower bound for numerical argument (min or inf)
    double m_dLarge;   // upper bound for numerical argument (max or sup)
    bool m_bMinSet;    // if true, compare:  small <= arg
    bool m_bMaxSet;    // if true, compare:  arg <= large
    bool m_bInfSet;    // if true, compare:  small < arg
    bool m_bSupSet;    // if true, compare:  arg < large

    // last error strings
    const char* m_acLastError;
    static char ms_acOptionNotFound[];
    static char ms_acArgumentRequired[];
    static char ms_acArgumentOutOfRange[];
    static char ms_acFilenameNotFound[];
};

}

#endif
