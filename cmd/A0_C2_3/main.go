package main

import (
	"fmt"

	"github.com/KyungWonPark/Correlation/internal/calc"
	"github.com/KyungWonPark/Correlation/internal/io"
	"github.com/gonum/matrix/mat64"
)

func main() { // subjectNum timeStart(300) timeEnd(900)
	numQueueSize := 4
	ringBuffer := make([]*mat64.Dense, numQueueSize)
	for i := 0; i < numQueueSize; i++ {
		ringBuffer[i] = mat64.NewDense(13362, 100, nil)
	}
	pl := calc.Init(numQueueSize, false)

	spv := io.NpytoMat64("newSpv.npy")
	c2 := mat64.NewDense(13362, 13362, nil)

	pl.Pearson(spv, c2)

	// io.Mat64toNpy(RESULTDIR+"/"+postFix+".npy", avgedMat)
	fmt.Println("Writing Results...")
	io.Mat64toNpy("c2.npy", c2)

	fmt.Println("Finished.")

	return
}
