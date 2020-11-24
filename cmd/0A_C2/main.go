package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/KyungWonPark/Correlation/internal/calc"
	"github.com/KyungWonPark/Correlation/internal/io"
	"github.com/gonum/matrix/mat64"
)

func main() { // subjectNum timeStart(300) timeEnd(900)
	subjectNum := os.Args[1]
	fmt.Printf("Processing: %s\n", subjectNum)
	timeStartStr := os.Args[2]
	timeEndStr := os.Args[3]
	timeStart, _ := strconv.Atoi(timeStartStr)
	timeEnd, _ := strconv.Atoi(timeEndStr)
	timePeriod := timeEnd - timeStart

	numQueueSize := 4
	ringBuffer := make([]*mat64.Dense, numQueueSize)
	for i := 0; i < numQueueSize; i++ {
		ringBuffer[i] = mat64.NewDense(13362, timePeriod, nil)
	}
	pl := calc.Init(numQueueSize, false)

	cgGray := mat64.NewDense(13362, timePeriod, nil)
	zscore := mat64.NewDense(13362, timePeriod, nil)
	spv := mat64.NewDense(13362, timePeriod, nil)
	c2 := mat64.NewDense(13362, 13362, nil)

	fileName := "./input/" + subjectNum + "_rfMRI_REST1_LR-smoothed.nii"

	fmt.Println("Calculating...")
	doSampling(fileName, timeStart, timeEnd, cgGray, 28)
	pl.ZScoring(cgGray, zscore)
	pl.Sigmoid(zscore, spv)
	pl.Pearson(spv, c2)

	outputDir := "./ouptut/" + subjectNum + "/" + timeStartStr + "-" + timeEndStr + "/"

	// io.Mat64toNpy(RESULTDIR+"/"+postFix+".npy", avgedMat)
	fmt.Println("Writing Results...")
	io.Mat64toNpy(outputDir+"bin/cgGray.npy", cgGray)
	io.Mat64toNpy(outputDir+"bin/zscore.npy", zscore)
	io.Mat64toNpy(outputDir+"bin/spv.npy", spv)
	io.Mat64toNpy(outputDir+"bin/c2.npy", c2)

	fmt.Println("Finished.")

	return
}
