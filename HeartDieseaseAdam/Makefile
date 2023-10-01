# Makefile for Writing Make Files Example

# *****************************************************
# Variables to control Makefile operation

CC = g++
CFLAGS = -Wall -g

# ****************************************************
# Targets needed to bring the executable up to date

main: main.o Neural.o
	$(CC) $(CFLAGS) -o main HeartDieseaseAdam.o Neural.o

# The main.o target can be written more simply

main.o: HeartDieseaseAdam.cpp Neural.h HeartDieseaseAdam.h
	$(CC) $(CFLAGS) -c HeartDieseaseAdam.cpp

Neural.o: Neural.h
