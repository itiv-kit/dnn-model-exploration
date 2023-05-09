result_checker: main.o sims_fpga.o sims.o
	gcc -o result_checker main.o sims_fpga.o sims.o
	

sims.o: sims.c
	gcc -c sims.c

sims_fpga.o: sims_fpga.c
	gcc -c sims_fpga.c

main.o: main.c
	gcc -c main.c


clean: 
	rm *.o result_checker