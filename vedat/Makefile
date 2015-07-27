# default programs
CC := cc

# output directory
O ?= .

# debug and optimization
OPT   ?= -O0
DEBUG ?= yes

ifeq ($(DEBUG),no)
	CPPFLAGS += -DNDEBUG
	CFLAGS   += -s
else
	CFLAGS   += -g
endif

# default flags
CPPFLAGS += -I.
CFLAGS   += -Wall -Wextra -fPIC -std=c99 -pipe $(OPT)
LDFLAGS  += -shared -lm

# source files, object files, executable
SRCS    := $(wildcard *.c)
OBJS    := $(addprefix $(O)/,$(SRCS:.c=.o))
LIBNAME := libwasp.so

# phony targets
.PHONY : all clean distclean

all: $(O)/$(LIBNAME)

clean:
	$(RM) $(O)/$(LIBNAME)
	$(RM) $(O)/*.o
	$(RM) $(O)/*.d

distclean: clean

# create build directory
$(O):
	mkdir -p $(O)

# link program
$(O)/$(LIBNAME): $(OBJS) | $(O)
	$(CC) -o $(O)/$(LIBNAME) $(CFLAGS) $(OBJS) $(LDFLAGS)

# pull in dependency info for *existing* .o files
-include $(OBJS:.o=.d)

# compile and generate dependency info
$(O)/%.o: %.c | $(O)
	$(CC) -c -o $(O)/$*.o -MMD -MP -MF $(O)/$*.d $(CPPFLAGS) $(CFLAGS) $*.c
