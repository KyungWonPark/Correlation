package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"strconv"

	"github.com/KyungWonPark/Correlation/internal/calc"
	"github.com/KyungWonPark/Correlation/internal/io"
	"github.com/ghetzel/shmtool/shm"
	"github.com/gonum/matrix/mat64"
	// "gonum.org/v1/gonum/blas/blas64"
	// blas_netlib "gonum.org/v1/netlib/blas/netlib"
)

func main() { // SUBJ TIMESTART TIMEEND anti-parallel GAMMA
	// blas64.Use(blas_netlib.Implementation{})
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
	inputRows, inputCols := c2.Dims()

	matBufferShm, err := shm.Create(inputRows * inputCols * 8)
	if err != nil {
		log.Fatalf("Failed to create shared memory region: %s\n", err)
	}

	eigValShm, err := shm.Create(inputRows * 1 * 8)
	if err != nil {
		log.Fatalf("Failed to create shared memory region: %s\n", err)
	}

	pMatBuffer, err := matBufferShm.Attach()
	if err != nil {
		log.Fatalf("Failed to attach shared memory region: %s\n", err)
	}

	pEigVal, err := eigValShm.Attach()
	if err != nil {
		log.Fatalf("Failed to attach shared memory region: %s\n", err)
	}

	arrThredMat := make([]float64, inputRows*inputCols)
	thredMat := mat64.NewDense(inputRows, inputCols, arrThredMat)
	arrEigVal := make([]float64, inputRows*1)
	eigVal := mat64.NewDense(inputRows, 1, arrEigVal)
	arrEigVec := make([]float64, inputRows*inputCols)
	eigVec := mat64.NewDense(inputRows, inputCols, arrEigVec)

	// Thresholds: Hard coded!
	thrs := []float64{0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
		0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96,
		0.97, 0.98, 0.99,
	}

	for _, thr := range thrs {
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

		pl.Laplacian(thredMat) // Now thredMat is a Laplacian matrix

		mat64tocArr(thredMat, pMatBuffer) // Copy thresholded matrix to MAGMA matrix buffer

		fmt.Printf("Diagonalizing...\n")
		// Call MAGMA
		// Fuck. I haredcoded. Fucking bitch cunt IBM LSF

		cmd := exec.Command("/home/iksoochang2/kw-park/.root/usr/local/bin/magma", fmt.Sprintf("%d", inputRows), fmt.Sprintf("%d", matBufferShm.Id), fmt.Sprintf("%d", eigValShm.Id))
		err := cmd.Run()
		if err != nil {
			log.Fatalf("[main.go - line 88] MAGMA execution has failed: %s\n", err)
		}

		// Copy result from MAGMA to eigVal and eigVec
		cArrtomat64(eigVal, pEigVal)
		cArrtomat64(eigVec, pMatBuffer) // eigVec rows are eigen vectors; eigVec[0] <- first eigen vector

		// output directory
		RESULTDIR := "output/" + subjStr + "/" + timeStartStr + "-" + timeEndStr + "/" + corrStr + "/" + "thr-" + fmt.Sprintf("%.2f", thr) + "/bin/"

		fmt.Println("Writing Eigen value")
		io.Mat64toNpy(RESULTDIR+"eigVal.npy", eigVal)
		fmt.Println("Writing Eigen vector")
		// io.Mat64toCSV(RESULTDIR+"/eigen-vector-thr-"+fmt.Sprintf("%f", thr)+".csv", eigVec)
		io.Mat64toNpy(RESULTDIR+"eigVec.npy", eigVec)

		fmt.Println("---- ---- ---- ---- ---- ---- ---- ----")
	}

	matBufferShm.Detach(pMatBuffer)
	matBufferShm.Destroy()

	eigValShm.Detach(pEigVal)
	eigValShm.Destroy()

	fmt.Println("Success.")
	return
}
