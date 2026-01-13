# CMU 15-418/618, Spring 2026

# Assignment 4

This is the starter code for Assignment 4 of CMU class 15-418/618, Spring 2026

Please review the course's policy on [academic
integrity](http://www.cs.cmu.edu/~418/academicintegrity.html),
regarding your obligation to keep your own solutions private from now
until eternity.

## Directory structure
```
├── code
│   ├── inputs
│   │   ├── easy_1024.txt
│   │   ├── hard_1024.txt
│   │   └── medium_1024.txt
│   ├── Makefile
│   ├── WireGrapher.java
│   ├── main.cpp
│   │      The starter code for reading and parsing command line arguments is provided.
│   ├── validate.py
│   │      Scripts to validate the consistency of output wire routes and cost array.
│   │      Run "python validate.py -h" to see the instructions to use the script
│   ├── wireroute.cpp
|   |      Implement your MPI code here
│   └── wireroute.h
├── examples: MPI handout
│   ├── Makefile
│   ├── main.c
│   ├── sqrt3.c
│   └── sqrt3.h
├── tutorials
│   └── PSC-machines.pdf
├── Makefile
└── README
```

## Get started
0. Read handout.
1. Read README.
2. Read `main.cpp`, `wireroute.cpp`, and `wireroute.h` including comments.
3. Move over any related code from Assignment 3.
4. Run using `mpirun -np <numprocs> <COMMAND>`. Remember to use `module load openmpi` when you get an interactive sesion on the PSC machines.
5. Answer questions on the handout.
