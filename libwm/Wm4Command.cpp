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

#include "Wm4FoundationPCH.h"
#include "Wm4Command.h"
using namespace Wm4;

char Command::ms_acOptionNotFound[] = "option not found";
char Command::ms_acArgumentRequired[] = "option requires an argument";
char Command::ms_acArgumentOutOfRange[] = "argument out of range";
char Command::ms_acFilenameNotFound[] = "filename not found";

//----------------------------------------------------------------------------
Command::Command (int iQuantity, char** apcArgument)
{
    m_iQuantity = iQuantity;
    m_acCmdline = 0;
    m_abUsed = 0;

    if (m_iQuantity > 0)
    {
        m_apcArgument = WM4_NEW char*[m_iQuantity];
        for (int i = 0; i < m_iQuantity; i++)
        {
            m_apcArgument[i] = apcArgument[i];
        }
    }
    else
    {
        m_apcArgument = 0;
    }

    Initialize();
}
//----------------------------------------------------------------------------
Command::Command (char* acCmdline)
{
    class Argument
    {
    public:
        char* m_pcItem;
        Argument* m_pkNext;
    };

    m_iQuantity = 0;
    m_apcArgument = 0;
    m_acCmdline = 0;
    m_abUsed = 0;

    if (acCmdline == 0 || strlen(acCmdline) == 0)
    {
        return;
    }

    size_t uiSize = strlen(acCmdline) + 1;
    m_acCmdline = WM4_NEW char[uiSize];
    System::Strcpy(m_acCmdline,uiSize,acCmdline);

    char* pcNextToken;
    char* pcToken = System::Strtok(m_acCmdline," \t",pcNextToken);
    Argument* pkList = 0;

    while (pcToken)
    {
        m_iQuantity++;

        Argument* pkCurrent = WM4_NEW Argument;
        pkCurrent->m_pcItem = pcToken;
        pkCurrent->m_pkNext = pkList;
        pkList = pkCurrent;

        pcToken = System::Strtok(0," \t",pcNextToken);
    }

    m_iQuantity++;
    m_apcArgument = WM4_NEW char*[m_iQuantity];
    m_apcArgument[0] = m_acCmdline;
    int i = m_iQuantity-1;
    while (pkList)
    {
        m_apcArgument[i--] = pkList->m_pcItem;

        Argument* pkSave = pkList->m_pkNext;
        WM4_DELETE pkList;
        pkList = pkSave;
    }

    Initialize();
}
//----------------------------------------------------------------------------
Command::~Command ()
{
    WM4_DELETE[] m_abUsed;
    WM4_DELETE[] m_apcArgument;
    WM4_DELETE[] m_acCmdline;
}
//----------------------------------------------------------------------------
void Command::Initialize ()
{
    m_abUsed = WM4_NEW bool[m_iQuantity];
    memset(m_abUsed,false,m_iQuantity*sizeof(bool));

    m_dSmall = 0.0;
    m_dLarge = 0.0;
    m_bMinSet = false;
    m_bMaxSet = false;
    m_bInfSet = false;
    m_bSupSet = false;

    m_acLastError = 0;
}
//----------------------------------------------------------------------------
int Command::ExcessArguments ()
{
    // checks to see if any command line arguments were not processed
    for (int i = 1; i < m_iQuantity; i++)
    {
        if (!m_abUsed[i])
        {
            return i;
        }
    }

    return 0;
}
//----------------------------------------------------------------------------
Command& Command::Min (double dValue)
{
    m_dSmall = dValue;
    m_bMinSet = true;
    return *this;
}
//----------------------------------------------------------------------------
Command& Command::Max (double dValue)
{
    m_dLarge = dValue;
    m_bMaxSet = true;
    return *this;
}
//----------------------------------------------------------------------------
Command& Command::Inf (double dValue)
{
    m_dSmall = dValue;
    m_bInfSet = true;
    return *this;
}
//----------------------------------------------------------------------------
Command& Command::Sup (double dValue)
{
    m_dLarge = dValue;
    m_bSupSet = true;
    return *this;
}
//----------------------------------------------------------------------------
int Command::Boolean (char* acName)
{
    bool bValue = false;
    return Boolean(acName,bValue);
}
//----------------------------------------------------------------------------
int Command::Boolean (char* acName, bool& rbValue)
{
    int iMatchFound = 0;
    rbValue = false;
    for (int i = 1; i < m_iQuantity; i++)
    {
        char* pcTmp = m_apcArgument[i];
        if (!m_abUsed[i] && pcTmp[0] == '-' && strcmp(acName,++pcTmp) == 0)
        {
            m_abUsed[i] = true;
            iMatchFound = i;
            rbValue = true;
            break;
        }
    }

    if (iMatchFound == 0)
    {
        m_acLastError = ms_acOptionNotFound;
    }

    return iMatchFound;
}
//----------------------------------------------------------------------------
int Command::Integer (char* acName, int& riValue)
{
    int iMatchFound = 0;
    for (int i = 1; i < m_iQuantity; i++)
    {
        char* pcTmp = m_apcArgument[i];
        if (!m_abUsed[i] && pcTmp[0] == '-' && strcmp(acName,++pcTmp) == 0)
        {
            // get argument
            pcTmp = m_apcArgument[i+1];
            if (m_abUsed[i+1] || (pcTmp[0] == '-' && !isdigit(pcTmp[1])))
            {
                m_acLastError = ms_acArgumentRequired;
                return 0;
            }
            riValue = atoi(pcTmp);
            if ((m_bMinSet && riValue < m_dSmall)
            ||  (m_bMaxSet && riValue > m_dLarge)
            ||  (m_bInfSet && riValue <= m_dSmall)
            ||  (m_bSupSet && riValue >= m_dLarge))
            {
                m_acLastError = ms_acArgumentOutOfRange;
                return 0;
            }
            m_abUsed[i] = true;
            m_abUsed[i+1] = true;
            iMatchFound = i;
            break;
        }
    }

    m_bMinSet = false;
    m_bMaxSet = false;
    m_bInfSet = false;
    m_bSupSet = false;

    if (iMatchFound == 0)
    {
        m_acLastError = ms_acOptionNotFound;
    }

    return iMatchFound;
}
//----------------------------------------------------------------------------
int Command::Float (char* acName, float& rfValue)
{
    int iMatchFound = 0;
    for (int i = 1; i < m_iQuantity; i++)
    {
        char* pcTmp = m_apcArgument[i];
        if (!m_abUsed[i] && pcTmp[0] == '-' && strcmp(acName,++pcTmp) == 0)
        {
            // get argument
            pcTmp = m_apcArgument[i+1];
            if (m_abUsed[i+1] || (pcTmp[0] == '-' && !isdigit(pcTmp[1])))
            {
                m_acLastError = ms_acArgumentRequired;
                return 0;
            }
            rfValue = (float)atof(pcTmp);
            if ((m_bMinSet && rfValue < m_dSmall)
            ||  (m_bMaxSet && rfValue > m_dLarge)
            ||  (m_bInfSet && rfValue <= m_dSmall)
            ||  (m_bSupSet && rfValue >= m_dLarge))
            {
                m_acLastError = ms_acArgumentOutOfRange;
                return 0;
            }
            m_abUsed[i] = true;
            m_abUsed[i+1] = true;
            iMatchFound = i;
            break;
        }
    }

    m_bMinSet = false;
    m_bMaxSet = false;
    m_bInfSet = false;
    m_bSupSet = false;

    if (iMatchFound == 0)
    {
        m_acLastError = ms_acOptionNotFound;
    }

    return iMatchFound;
}
//----------------------------------------------------------------------------
int Command::Double (char* acName, double& rdValue)
{
    int iMatchFound = 0;
    for (int i = 1; i < m_iQuantity; i++)
    {
        char* pcTmp = m_apcArgument[i];
        if (!m_abUsed[i] && pcTmp[0] == '-' && strcmp(acName,++pcTmp) == 0)
        {
            // get argument
            pcTmp = m_apcArgument[i+1];
            if (m_abUsed[i+1] || (pcTmp[0] == '-' && !isdigit(pcTmp[1])))
            {
                m_acLastError = ms_acArgumentRequired;
                return 0;
            }
            rdValue = atof(pcTmp);
            if ((m_bMinSet && rdValue < m_dSmall)
            ||  (m_bMaxSet && rdValue > m_dLarge)
            ||  (m_bInfSet && rdValue <= m_dSmall)
            ||  (m_bSupSet && rdValue >= m_dLarge))
            {
                m_acLastError = ms_acArgumentOutOfRange;
                return 0;
            }
            m_abUsed[i] = true;
            m_abUsed[i+1] = true;
            iMatchFound = i;
            break;
        }
    }

    m_bMinSet = false;
    m_bMaxSet = false;
    m_bInfSet = false;
    m_bSupSet = false;

    if (iMatchFound == 0)
    {
        m_acLastError = ms_acOptionNotFound;
    }

    return iMatchFound;
}
//----------------------------------------------------------------------------
int Command::String (char* acName, char*& racValue)
{
    int iMatchFound = 0;
    for (int i = 1; i < m_iQuantity; i++)
    {
        char* pcTmp = m_apcArgument[i];
        if (!m_abUsed[i] && pcTmp[0] == '-' && strcmp(acName,++pcTmp) == 0)
        {
            // get argument
            pcTmp = m_apcArgument[i+1];
            if (m_abUsed[i+1] || pcTmp[0] == '-')
            {
                m_acLastError = ms_acArgumentRequired;
                return 0;
            }

            size_t uiSize = strlen(pcTmp) + 1;
            racValue = WM4_NEW char[uiSize];
            System::Strcpy(racValue,uiSize,pcTmp);
            m_abUsed[i] = true;
            m_abUsed[i+1] = true;
            iMatchFound = i;
            break;
        }
    }

    if (iMatchFound == 0)
    {
        m_acLastError = ms_acOptionNotFound;
    }

    return iMatchFound;
}
//----------------------------------------------------------------------------
int Command::Filename (char*& racName)
{
    int iMatchFound = 0;
    for (int i = 1; i < m_iQuantity; i++)
    {
        char* pcTmp = m_apcArgument[i];
        if (!m_abUsed[i] && pcTmp[0] != '-')
        {
            size_t uiSize = strlen(pcTmp) + 1;
            racName = WM4_NEW char[uiSize];
            System::Strcpy(racName,uiSize,pcTmp);
            m_abUsed[i] = true;
            iMatchFound = i;
            break;
        }
    }

    if (iMatchFound == 0)
    {
        m_acLastError = ms_acFilenameNotFound;
    }

    return iMatchFound;
}
//----------------------------------------------------------------------------
const char* Command::GetLastError ()
{
    return m_acLastError;
}
//----------------------------------------------------------------------------
