package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"unsafe"

	"github.com/KyungWonPark/Correlation/internal/calc"
	"github.com/KyungWonPark/Correlation/internal/io"
	"github.com/ghetzel/shmtool/shm"
	"github.com/gonum/matrix/mat64"
)

func main() {

	DATADIR := os.Getenv("DATA")
	RESULTDIR := os.Getenv("RESULT")

	numQueueSize := 8
	ringBuffer := make([]*mat64.Dense, numQueueSize)
	for i := 0; i < numQueueSize; i++ {
		ringBuffer[i] = mat64.NewDense(13362, 600, nil)
	}

	pl := calc.Init(numQueueSize, true)

	go func() {
		for _, file := range fileList {
			dest := pl.Malloc()
			doSampling(DATADIR+"/fMRI-Smoothed/"+file, ringBuffer[dest], pl.GetNP())
			pl.Push(dest)
		}

		pl.Close()
		pl.StopScheduler()

		return
	}()

	avgedMat := mat64.NewDense(13362, 13362, nil)

	{
		accedMat := mat64.NewDense(13362, 13362, nil)
		{
			temp0 := mat64.NewDense(13362, 600, nil)
			temp1 := mat64.NewDense(13362, 600, nil)
			temp2 := mat64.NewDense(13362, 13362, nil)

			for {
				job, ok := pl.Pop()
				if ok {
					pl.ZScoring(ringBuffer[job], temp0)
					pl.Sigmoid(temp0, temp1)
					pl.Pearson(temp1, temp2)
					pl.Acc(temp2, accedMat)

					pl.Free(job)
				} else {
					break
				}
			}
		}
		pl.Avg(accedMat, avgedMat, float64(len(fileList)))
	}

	thredShm, err := shm.Create(13362 * 13362 * 8)
	if err != nil {
		log.Fatal("Failed to allocate shared memory region.", err)
	}
	eigValShm, err := shm.Create(13362 * 1 * 8)
	if err != nil {
		log.Fatal("Failed to allocate shared memory region.", err)
	}
	eigVecShm, err := shm.Create(13362 * 13362 * 8)
	if err != nil {
		log.Fatal("Failed to allocate shared memory region.", err)
	}

	thredBase, err := thredShm.Attach()
	if err != nil {
		log.Fatal("Failed to attach shared memory region.", err)
	}
	eigValBase, err := thredShm.Attach()
	if err != nil {
		log.Fatal("Failed to attach shared memory region.", err)
	}
	eigVecBase, err := thredShm.Attach()
	if err != nil {
		log.Fatal("Failed to attach shared memory region.", err)
	}

	thredBackingArr := (*[13362 * 13362]float64)(unsafe.Pointer(uintptr(thredBase)))
	eigValBackingArr := (*[13362 * 1]float64)(unsafe.Pointer(uintptr(eigValBase)))
	eigVecBackingArr := (*[13362 * 13362]float64)(unsafe.Pointer(uintptr(eigVecBase)))

	thredMat := mat64.NewDense(13362, 13362, (*thredBackingArr)[:])
	eigVal := mat64.NewDense(13362, 1, (*eigValBackingArr)[:])
	eigVec := mat64.NewDense(13362, 13362, (*eigVecBackingArr)[:])

	var thr float64
	for thr = 0; thr < 1; thr += 0.5 {
		pl.Threshold(avgedMat, thredMat, thr)

		// Call MAGMA routine
		magmaCmd := exec.Command("./files/magma", "13362", fmt.Sprintf("%d", thredShm.Id), fmt.Sprintf("%d", eigValShm.Id), fmt.Sprintf("%d", eigVecShm.Id))
		_, err := magmaCmd.Output()
		if err != nil {
			log.Fatal("Failed to execute MAGMA routine.", err)
		}

		io.Mat64toCSV(RESULTDIR+"/c2-thr-"+fmt.Sprintf("%f", thr)+".csv", thredMat)
		io.Mat64toCSV(RESULTDIR+"/eigVal-thr-"+fmt.Sprintf("%f", thr)+".csv", eigVal)
		io.Mat64toCSV(RESULTDIR+"/eigVec-thr-"+fmt.Sprintf("%f", thr)+".csv", eigVec)
	}

	thredShm.Detach(thredBase)
	eigValShm.Detach(eigValBase)
	eigVecShm.Detach(eigVecBase)

	thredShm.Destroy()
	eigValShm.Destroy()
	eigVecShm.Destroy()

	return
}
