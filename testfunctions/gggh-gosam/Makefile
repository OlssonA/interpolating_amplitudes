#-*- Makefile -*-
COMPILER=gfortran

OBJ=obj-$(COMPILER)
OBJDIR:=$(OBJ)

F77=gfortran -ffixed-line-length-none -J$(OBJ) -I$(OBJ)
## -fbounds-check sometimes causes a weird error due to non-lazy evaluation
## of boolean in gfortran.
#FFLAGS= -Wall -Wimplicit-interface -fbounds-check
## For floating point exception trapping  uncomment the following 
#FPE=-ffpe-trap=invalid,zero,overflow,underflow 
## gfortran 4.4.1 optimized with -O3 yields erroneous results
## Use -O2 to be on the safe side
OPT=-O2
## For debugging uncomment the following
#DEBUG= -ggdb 
ifdef DEBUG
OPT=-O0
FPE=-ffpe-trap=invalid,zero,overflow
#,underflow
endif

GOSAMLIBS=$(shell cd GoSamlib ; echo *.f *.f90 ' ' | sed 's/qlonshellcutoff.f// ; s/qlconstants.f// ; s/.f /.o /g ; s/.f90 /.o /g ; ' | tee GoSamlib.txt)
# GOSAMLIBS+=$(shell cd GoSamlib ; echo *.cpp *.cc *.F90 ' ' | sed 's/.cpp /.o /g ; s/.cc /.o /g ; s/.F90 /.o /g ; ' | tee -a GoSamlib.txt)
GOSAMLIBS+=$(shell cd GoSamlib ; echo *.cc *.F90 ' ' | sed 's/.cc /.o /g ; s/.F90 /.o /g ; ' | tee -a GoSamlib.txt)
GOSAMLIBS+=$(shell cd GoSamlib/quadsources ; echo *.cc *.F90 ' ' | sed 's/.cc /.o /g ; s/.F90 /.o /g ; ' | tee -a ../GoSamlib.txt)

PWD=$(shell pwd)
WDNAME=$(shell basename $(PWD))
VPATH= ./:../:$(OBJDIR):./GoSamlib/:./Gosamlib/quadsources/

INCLUDE0=$(PWD)
INCLUDE1=$(PWD)/GoSamlib
FF=$(F77) $(FFLAGS) $(FPE) $(OPT) $(DEBUG) -I$(INCLUDE0) -I$(INCLUDE1)

INCLUDE =
#$(wildcard ../include/*.h *.h include/*.h)

NINJAFLAGS=-DHAVE_CONFIG_H -I./GoSamlib -I./GoSamlib/ninja -O2 -fcx-limited-range
NINJAFLAGSQP=-DHAVE_CONFIG_H -I./GoSamlib -I./Gosamlib/quadsources -O2

%.o: %.f $(INCLUDE) | $(OBJDIR)
	$(FF) -c -o $(OBJ)/$@ $<

%.o: %.f90 $(INCLUDE) | $(OBJDIR)
	$(FF) -c -o $(OBJ)/$@ $<

%.o: %.F90 $(INCLUDE) | $(OBJDIR)
	$(FF) -DHAVE_CONFIG_H -c -o $(OBJ)/$@ $<

%.o: quadsources/%.F90 $(INCLUDE) | $(OBJDIR)
	$(FF) -DHAVE_CONFIG_H -c -o $(OBJ)/$@ $<

%.o: %.c | $(OBJDIR)
	$(CC) $(DEBUG) -c -o $(OBJ)/$@ $^ 

%.o: %.cpp
	$(CXX) $(DEBUG) -c -o $(OBJ)/$@ $^

%.o: %.cc | $(OBJDIR)
	$(CXX) $(DEBUG) -c -o $(OBJ)/$@ $^ $(NINJAFLAGS) $(FJCXXFLAGS)

%.o: quadsources/%.cc | $(OBJDIR)
	$(CXX) $(DEBUG) -c -o $(OBJ)/$@ $^ $(NINJAFLAGSQP) $(FJCXXFLAGS)

LIBS+=-lz -lquadmath


gosamlib.a: $(GOSAMLIBS)
	@rm -f $(OBJ)/gosamlib.a
	$(PWD)/GoSamlib/compile_gosamlib.sh $(PWD)/$(OBJ)
	mv $(OBJ)/gosamlib.a $@


clean:
	rm -rf obj-gfortran


veryclean: clean
	rm -f $(OBJ)/*.o $(OBJ)/*.mod $(OBJ)/*.a GoSamlib.lst $(PWD)/GoSamlib/GoSamlib.txt gosamlib.a

# target to generate object directory if it does not exist
$(OBJDIR):
	mkdir -p $(OBJDIR)

##########################################################################

include GoSamlib/Makefile.virt.dep


