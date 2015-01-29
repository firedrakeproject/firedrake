# ************************************************************************
# * Copyright 2010 Paulo A. Herrera. All rights reserved.                           *
# *                                                                                 *
# * Redistribution and use in source and binary forms, with or without              *
# * modification, are permitted provided that the following conditions are met:     *
# *                                                                                 *
# *  1. Redistributions of source code must retain the above copyright notice,      *
# *  this list of conditions and the following disclaimer.                          *
# *                                                                                 *
# *  2. Redistributions in binary form must reproduce the above copyright notice,   *
# *  this list of conditions and the following disclaimer in the documentation      *
# *  and/or other materials provided with the distribution.                         *
# *                                                                                 *
# * THIS SOFTWARE IS PROVIDED BY PAULO A. HERRERA ``AS IS'' AND ANY EXPRESS OR      *
# * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF    *
# * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO      *
# * EVENT SHALL <COPYRIGHT HOLDER> OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,        *
# * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,  *
# * BUT NOT LIMITED TO, PROCUREMEN OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    *
# * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY           *
# * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING  *
# * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS              *
# * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                    *
# ************************************************************************

# **************************************
# *  Simple class to generate a        *
# *  well-formed XML file.             *
# **************************************


class XmlWriter:

    def __init__(self, filepath, addDeclaration=True):
        self.stream = open(filepath, "w")
        self.openTag = False
        self.current = []
        if (addDeclaration):
            self.addDeclaration()

    def close(self):
        assert(not self.openTag)
        self.stream.close()

    def addDeclaration(self):
        self.stream.write('<?xml version="1.0"?>')

    def openElement(self, tag):
        if self.openTag:
            self.stream.write(">")
        self.stream.write("\n<%s" % tag)
        self.openTag = True
        self.current.append(tag)
        return self

    def closeElement(self, tag=None):
        if tag:
            assert(self.current.pop() == tag)
            if (self.openTag):
                self.stream.write(">")
                self.openTag = False
            self.stream.write("\n</%s>" % tag)
        else:
            self.stream.write("/>")
            self.openTag = False
            self.current.pop()
        return self

    def addText(self, text):
        if (self.openTag):
            self.stream.write(">\n")
            self.openTag = False
        self.stream.write(text)
        return self

    def addAttributes(self, **kwargs):
        assert (self.openTag)
        for key in kwargs:
            self.stream.write(' %s="%s"' % (key, kwargs[key]))
        return self
