#!/bin/sh
#### Requesting job resources

#### Specifying a job name
### BSUB -J [jobname]
#BSUB -J NEWJOB

#### CPU slot [N]
#### N=[core number]
#BSUB -n 32

####node per cpu
####BSUB -R "span[ptile=2]"

#### GPU slot [gpunum]
#### GPU num=[gpu number]
#BSUB -gpu num=0

#### Setting email recipient list
### entry do not send mail
#BSUB -u park0kyung0won@dgist.ac.kr

#### Specifying queue and/or server
#BSUB -q normal

#### Marking a job as re-runnable or not
#BSUB -r n

#### Create Stdout.out File 
#BSUB -o /home/iksoochang2/kw-park/Log/%J_outfile

#### Create ERROR Stdout.out File 
#BSUB -e /home/iksoochang2/kw-park/Log/%J_outfile

#### mpirun config
#BSUB -a openmpi 

#### JOB COMMAND

# Environment Variable Loading
/bin/bash /home/iksoochang2/kw-park/.bashrc

# Command
make -j32
