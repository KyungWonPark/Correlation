package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/KyungWonPark/Correlation/internal/io"
	"github.com/KyungWonPark/nifti"
	"github.com/gonum/matrix/mat64"
)

// Voxel struct represent a voxel
type Voxel struct {
	x uint32
	y uint32
	z uint32
}

var greyListX [][]Voxel
var greyListY [][]Voxel
var greyListZ [][]Voxel

func init() {
	greyListX = make([][]Voxel, 71)
	greyListY = make([][]Voxel, 88)
	greyListZ = make([][]Voxel, 67)

	// Read grey voxel list into greyList
	f, err := os.Open("bin/greyVoxels.dat")
	if err != nil {
		log.Fatal("Failed to open greyVoxels.dat file!")
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		xyz := strings.Split(line, ",")
		x, err0 := strconv.Atoi(xyz[0])
		y, err1 := strconv.Atoi(xyz[1])
		z, err2 := strconv.Atoi(xyz[2])
		if err0 != nil || err1 != nil || err2 != nil {
			log.Fatal("Failed to convert ascii to integer!", err0)
		}

		v := Voxel{uint32(x), uint32(y), uint32(z)}
		greyListX[x-11] = append(greyListX[x-11], v)
		greyListY[y-12] = append(greyListY[y-12], v)
		greyListZ[z-12] = append(greyListZ[z-12], v)
	}

	fmt.Println("Finished reading voxelGray.dat")
	return
}

func main() {
	// Open nitfi image
	var img nifti.Nifti1Image
	filePath := "bin/165941_rfMRI_REST1_LR-smoothed.nii"
	img.LoadImage(filePath, true)

	// Do layer-coarse-graining
	spvSumX, spvAvgX := layerCoarseGraining(&img, "X")
	spvSumY, spvAvgY := layerCoarseGraining(&img, "Y")
	spvSumZ, spvAvgZ := layerCoarseGraining(&img, "Z")

	io.Mat64toNpy("bin/spvSumX.npy", spvSumX)
	io.Mat64toNpy("bin/spvAvgX.npy", spvAvgX)
	io.Mat64toNpy("bin/spvSumY.npy", spvSumY)
	io.Mat64toNpy("bin/spvAvgY.npy", spvAvgY)
	io.Mat64toNpy("bin/spvSumZ.npy", spvSumZ)
	io.Mat64toNpy("bin/spvAvgZ.npy", spvAvgZ)

	return
}

func layerCoarseGraining(img *nifti.Nifti1Image, axis string) (*mat64.Dense, *mat64.Dense) {
	timeStart := 300
	timeEnd := 900
	timePoints := timeEnd - timeStart

	layersCnt := -1
	var indices [][]Voxel

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
			for t := timeStart; t < timeEnd; t++ {
				acc := 0.0
				cnt := 0
				for _, v := range indices[i] {
					acc += float64(img.GetAt(v.x, v.y, v.z, uint32(t)))
					cnt++
				}

				avg := (acc / float64(cnt))
				spvSum.Set(i, t-timeStart, acc)
				spvAvg.Set(i, t-timeStart, avg)
			}

			wg.Done()
			return
		}(i)
	}
	wg.Wait()

	return spvSum, spvAvg
}
