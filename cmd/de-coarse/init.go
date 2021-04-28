package main

import (
	"bufio"
	"log"
	"os"
	"strconv"
	"strings"
)

type VoxelCoord struct {
	x      int
	y      int
	z      int
	isGray bool
	value  float64
}

type Voxel struct {
	voxType int
	value   float64
	denom   int
}

var seedList [13362]VoxelCoord
var fineMap [91][109][91]Voxel

var shareMap [4]int

func init() {
	shareMap[0] = 8
	shareMap[1] = 4
	shareMap[2] = 2
	shareMap[3] = 1

	f, err := os.Open("greyList.txt")
	if err != nil {
		log.Fatal("Failed to open file: greyList.txt", err)
	}
	defer f.Close()

	i := 0
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.Split(scanner.Text(), ",")
		x, err0 := strconv.Atoi(line[0])
		y, err1 := strconv.Atoi(line[1])
		z, err2 := strconv.Atoi(line[2])
		if err0 != nil || err1 != nil || err2 != nil {
			log.Fatal("Failed to convert ascii to integer", err0, err1, err2)
		}

		seedList[i].x = 2*x - 1
		seedList[i].y = 2*y - 1
		seedList[i].z = 2 * z
		seedList[i].isGray = true

		i++
	}

	g, err := os.Open("TD_type.dat")
	if err != nil {
		log.Fatal("Failed to open file: TD_type.dat", err)
	}
	defer g.Close()

	scanner = bufio.NewScanner(g)
	for scanner.Scan() {
		line := strings.Split(scanner.Text(), " ")
		x, err0 := strconv.Atoi(line[0])
		y, err1 := strconv.Atoi(line[1])
		z, err2 := strconv.Atoi(line[2])
		if err0 != nil || err1 != nil || err2 != nil {
			log.Fatal("Failed to convert ascii to integer", err0, err1, err2)
		}

		if line[3] == "1.000" { // Grey Voxel
			fineMap[x-1][y-1][z-1].voxType = 1
		} else if line[3] == "2.000" { // White Voxel
			fineMap[x-1][y-1][z-1].voxType = 2
			fineMap[x-1][y-1][z-1].value = 0
		} else { // Void
			fineMap[x-1][y-1][z-1].value = -2000
		}
	}
}
