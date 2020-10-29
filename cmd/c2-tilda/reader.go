package main

import (
	"C"
	"bufio"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/KyungWonPark/nifti"
	"github.com/gonum/matrix/mat64"
)

// Voxel represents fMRI voxel coordinates
type Voxel struct {
	x int
	y int
	z int
}

var fileList []string
var convKernel [3][3][3]float64
var greyVoxels [13362]Voxel

func init() {
	// Sampling setting
	for z := 0; z < 2; z++ {
		for y := 0; y < 2; y++ {
			for x := 0; x < 2; x++ {
				taxiDist := math.Abs(float64(x-1)) + math.Abs(float64(y-1)) + math.Abs(float64(z-1))
				convKernel[z][y][x] = math.Pow(2, -1*taxiDist)
			}
		}
	}

	f, err := os.Open("files/greyList.txt")
	if err != nil {
		log.Fatal("Failed to open greyList.txt file!", err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	i := 0
	for scanner.Scan() {
		line := scanner.Text()
		xyz := strings.Split(line, ",")
		x, err0 := strconv.Atoi(xyz[0])
		y, err1 := strconv.Atoi(xyz[1])
		z, err2 := strconv.Atoi(xyz[2])
		if err0 != nil || err1 != nil || err2 != nil {
			log.Fatal("Failed to convert ascii to integer!", err0)
		}

		greyVoxels[i] = Voxel{x, y, z}
		i++
	}

	g, err := os.Open("files/fileList.txt")
	if err != nil {
		log.Fatal("Failed to open fileList.txt!", err)
	}
	defer g.Close()

	scanner = bufio.NewScanner(g)
	for scanner.Scan() {
		line := scanner.Text()
		fileList = append(fileList, line)
	}

	return
}

func convolution(img *nifti.Nifti1Image, timePoint int, seed Voxel) float64 {
	var value float64

	for k := -1; k < 2; k++ {
		for j := -1; j < 2; j++ {
			for i := -1; i < 2; i++ {
				value += float64(img.GetAt(uint32(seed.x+i), uint32(seed.y+j), uint32(seed.z+k), uint32(timePoint))) * convKernel[k+1][j+1][i+1]
			}
		}
	}

	return value / 8
}

func sampling(img *nifti.Nifti1Image, order <-chan int, wg *sync.WaitGroup, timeSeries *mat64.Dense) {
	for {
		timePoint, ok := <-order
		if ok {
			for i, vox := range greyVoxels {
				seed := Voxel{2 * vox.x, 2 * vox.y, 1 + 2*vox.z}
				timeSeries.Set(i, timePoint-300, convolution(img, timePoint, seed))
			}
			wg.Done()
		} else {
			break
		}
	}

	return
}

func doSampling(path string, timeSeries *mat64.Dense, numLoader int) {
	var img nifti.Nifti1Image
	img.LoadImage(path, true)

	order := make(chan int, numLoader)
	var wg sync.WaitGroup

	wg.Add(600)
	for i := 0; i < numLoader; i++ {
		go sampling(&img, order, &wg, timeSeries)
	}

	for timePoint := 300; timePoint < 900; timePoint++ {
		order <- timePoint
	}
	wg.Wait()

	close(order)
	return
}
