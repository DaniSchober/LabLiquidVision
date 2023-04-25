CPP = g++
CPPFLAGS = -Wall -O3 -DNDEBUG # -g
LDFLAGS = -lm

all: libmesh_query.a

libmesh_query.a: mesh_query.o predicates.o bounding_box_tree.o
	ar r $@ mesh_query.o predicates.o bounding_box_tree.o;
	ranlib $@

mesh_query.o: mesh_query.cpp mesh_query.h bounding_box_tree.h bounding_box.h predicates.h vec.h util.h
	$(CPP) $(CPPFLAGS) -o $@ -c mesh_query.cpp

predicates.o: predicates.cpp predicates.h
	$(CPP) $(CPPFLAGS) -o $@ -c predicates.cpp

bounding_box_tree.o: bounding_box_tree.cpp bounding_box_tree.h bounding_box.h vec.h util.h
	$(CPP) $(CPPFLAGS) -o $@ -c bounding_box_tree.cpp

clean:
	-rm libmesh_query.a mesh_query.o predicates.o bounding_box_tree.o
