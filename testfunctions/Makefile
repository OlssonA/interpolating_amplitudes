.DEFAULT_GOAL: all 

.PHONY:all

all: gggh-eval tth-eval

tth-eval: tth-eval.cpp tth-gosam/gosamlib.a
	${CXX} -o $@.tmp ${CXXFLAGS} -Wall -Wno-unused-variable -O3 -g tth-eval.cpp ${LDFLAGS} tth-gosam/gosamlib.a -lz -lm -lquadmath -lgfortran
	mv $@.tmp $@

gggh-eval: gggh-eval.cpp gggh-gosam/gosamlib.a
	${CXX} -o $@.tmp ${CXXFLAGS} -Wall -Wno-unused-variable -O3 -g gggh-eval.cpp ${LDFLAGS} gggh-gosam/gosamlib.a -lz -lm -lquadmath -lgfortran
	mv $@.tmp $@

tth-gosam/gosamlib.a:
	+$(MAKE) -C tth-gosam gosamlib.a

gggh-gosam/gosamlib.a:
	+$(MAKE) -C gggh-gosam gosamlib.a

clean:
	rm -f tth-eval.tmp tth-eval gggh-eval.tmp gggh-eval
