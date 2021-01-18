package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/KyungWonPark/Correlation/internal/calc"
	"github.com/KyungWonPark/Correlation/internal/io"
	"github.com/KyungWonPark/nifti"
	"github.com/gonum/matrix/mat64"
)

// Voxel struct represent a voxel
type Voxel struct {
	x int
	y int
	z int
}

var greyListX [][]int
var greyListY [][]int
var greyListZ [][]int

var greyIndex [67855]Voxel

func init() {
	greyListX = make([][]int, 71)
	greyListY = make([][]int, 88)
	greyListZ = make([][]int, 67)

	// Read grey voxel list into greyList
	f, err := os.Open("bin/greyVoxels.dat")
	if err != nil {
		log.Fatal("Failed to open greyVoxels.dat file!")
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		xyz := strings.Split(line, " ")
		voxelIdx, err3 := strconv.Atoi(xyz[0])
		x, err0 := strconv.Atoi(xyz[1])
		y, err1 := strconv.Atoi(xyz[2])
		z, err2 := strconv.Atoi(xyz[3])
		if err0 != nil || err1 != nil || err2 != nil || err3 != nil {
			log.Fatal("Failed to convert ascii to integer!", err0)
		}

		greyListX[x-11] = append(greyListX[x-11], voxelIdx-1)
		greyListY[y-12] = append(greyListY[y-12], voxelIdx-1)
		greyListZ[z-12] = append(greyListZ[z-12], voxelIdx-1)

		v := Voxel{x, y, z}
		greyIndex[voxelIdx-1] = v
	}

	fmt.Println("Finished reading greyVoxels.dat")
	return
}

func main() {
	// Open nitfi image
	var img nifti.Nifti1Image
	filePath := "bin/165941_rfMRI_REST1_LR-smoothed.nii"
	img.LoadImage(filePath, true)

	timeStart := 300
	timeEnd := 900
	timePoints := timeEnd - timeStart

	// Turn NIFTI image to timeSeries (mat64.Dense)
	cgGray := mat64.NewDense(67855, timePoints, nil)

	var wg sync.WaitGroup
	for i := 0; i < 67855; i++ {
		wg.Add(1)
		go func(i int) {
			for t := timeStart; t < timeEnd; t++ {
				Vox := greyIndex[i]

				value := img.GetAt(uint32(Vox.x), uint32(Vox.y), uint32(Vox.z), uint32(t))
				cgGray.Set(i, t-timeStart, float64(value))
			}

			wg.Done()
			return
		}(i)
	}
	wg.Wait()

	io.Mat64toNpy("bin/cgGray.npy", cgGray)

	// Z-Scoring and Sigmoidal filter
	numQueueSize := 4
	ringBuffer := make([]*mat64.Dense, numQueueSize)
	for i := 0; i < numQueueSize; i++ {
		ringBuffer[i] = mat64.NewDense(13362, timePoints, nil)
	}
	pl := calc.Init(numQueueSize, false)

	zScored := mat64.NewDense(67855, timePoints, nil)
	pl.ZScoring(cgGray, zScored)
	io.Mat64toNpy("bin/zScore.npy", zScored)

	spv := mat64.NewDense(67855, timePoints, nil)
	pl.Sigmoid(zScored, spv)
	io.Mat64toNpy("bin/spv.npy", spv)

	// Do layer-coarse-graining
	spvSumX, spvAvgX := layerCoarseGraining(spv, "X")
	spvSumY, spvAvgY := layerCoarseGraining(spv, "Y")
	spvSumZ, spvAvgZ := layerCoarseGraining(spv, "Z")

	io.Mat64toNpy("bin/spvSumX.npy", spvSumX)
	io.Mat64toNpy("bin/spvAvgX.npy", spvAvgX)
	io.Mat64toNpy("bin/spvSumY.npy", spvSumY)
	io.Mat64toNpy("bin/spvAvgY.npy", spvAvgY)
	io.Mat64toNpy("bin/spvSumZ.npy", spvSumZ)
	io.Mat64toNpy("bin/spvAvgZ.npy", spvAvgZ)

	return
}

func layerCoarseGraining(timeSeries *mat64.Dense, axis string) (*mat64.Dense, *mat64.Dense) {
	timeStart := 300
	timeEnd := 900
	timePoints := timeEnd - timeStart

	layersCnt := -1
	var indices [][]int

	if axis == "X" {
		layersCnt = 71
		indices = greyListX
	} else if axis == "Y" {
		layersCnt = 88
		indices = greyListY
	} else if axis == "Z" {
		layersCnt = 67
		indices = greyListZ
	} else {
		log.Fatal("No such axis!")
	}

	spvSum := mat64.NewDense(layersCnt, timePoints, nil)
	spvAvg := mat64.NewDense(layersCnt, timePoints, nil)

	var wg sync.WaitGroup
	for i := 0; i < layersCnt; i++ {
		wg.Add(1)
		go func(i int) {
			for t := 0; t < timePoints; t++ {
				acc := 0.0
				cnt := 0
				for _, v := range indices[i] {
					acc += timeSeries.At(v, t)
					cnt++
				}

				avg := (acc / float64(cnt))
				spvSum.Set(i, t, acc)
				spvAvg.Set(i, t, avg)
			}

			wg.Done()
			return
		}(i)
	}
	wg.Wait()

	return spvSum, spvAvg
}
