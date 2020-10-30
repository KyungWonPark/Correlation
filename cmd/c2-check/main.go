package main

import (
	"fmt"
	"log"
	"os"

	"github.com/gonum/matrix/mat64"
)

func main() { // thrStart thrEnd thrItv isDebugMode
	DATADIR := os.Getenv("DATA")
	RESULTDIR := os.Getenv("RESULT")
	RESULTDIR = RESULTDIR + "/check"

	timeSeries := doSampling(DATADIR + "/100610_rfMRI_REST1_LR-smoothed.nii")
	write(RESULTDIR+"/100610_CoarseGrained.txt", timeSeries)

	return
}

func write(path string, timeSeries *mat64.Dense) {
	_, cols := timeSeries.Dims()

	f, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	for t := 0; t < cols; t++ { // time
		line := fmt.Sprintf("timepoint %d\n", (t + 1))
		_, err := f.WriteString(line)
		if err != nil {
			log.Fatal(err)
		}
		for j := 0; j < len(greyVoxels); j++ {
			x := greyVoxels[j].x
			y := greyVoxels[j].y
			z := greyVoxels[j].z

			line := fmt.Sprintf("%d %d %d %.3f\n", x, y, z, timeSeries.At(j, t))
			_, err := f.WriteString(line)
			if err != nil {
				log.Fatal(err)
			}
		}
	}
	return
}
