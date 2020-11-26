package main

import (
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/KyungWonPark/Correlation/internal/calc"
	"github.com/KyungWonPark/Correlation/internal/io"
	"github.com/gonum/matrix/mat64"
)

func main() { // SUBJ TIMESTART TIMEEND anti-parallel GAMMA
	subjStr := os.Args[1]
	timeStartStr := os.Args[2]
	timeEndStr := os.Args[3]
	corrStr := os.Args[4]
	gamma, _ := strconv.ParseFloat(os.Args[5], 64)

	pl := calc.Init(4, false)

	fmt.Println("Reading C2...")
	c2Path := "input/" + subjStr + "/" + timeStartStr + "-" + timeEndStr + "/" + "bin/"

	// io.Mat64toCSV(RESULTDIR+"/c2-tilda.csv", avgedMat)
	c2 := io.NpytoMat64(c2Path + "c2.npy")

	arrThredMat := make([]float64, 13362*13362)
	thredMat := mat64.NewDense(13362, 13362, arrThredMat)

	// Thresholds: Hard coded!
	thrs := []float64{0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
		0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96,
		0.97, 0.98, 0.99,
	}

	for _, thr := range thrs {
		RESULTDIR := "output/" + subjStr + "/" + timeStartStr + "-" + timeEndStr + "/" + corrStr + "/" + "thr-" + fmt.Sprintf("%.2f", thr) + "/bin/"

		fmt.Printf("Processing: thr - %.2f\n", thr)

		if corrStr == "parallel" {
			pl.Threshold(c2, thredMat, thr, gamma)
		} else if corrStr == "anti-parallel" {
			pl.InvThreshold(c2, thredMat, thr, gamma)
		} else if corrStr == "any-parallel" {
			pl.AbsThreshold(c2, thredMat, thr, gamma)
		} else {
			log.Fatalln("No such correlation type, aboring.")
		}

		io.Mat64toNpy(RESULTDIR+"c2-thresholded.npy", thredMat)
	}

	fmt.Println("Success.")
	return
}
