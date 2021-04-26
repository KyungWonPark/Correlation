package main

import (
	"fmt"
	"log"
	"os"
	"runtime"
	"sync"

	"github.com/KyungWonPark/nifti"
	"github.com/kshedden/gonpy"
)

func main() {
	fileName := os.Args[1]
	fmt.Printf("Filename: %s\n", fileName)

	npyReader, err := gonpy.NewFileReader(fileName)
	if err != nil {
		log.Fatal("Cannot open npy file", err)
	}

	npyDim := npyReader.Shape[0]
	npyDat, err := npyReader.GetFloat64()
	if err != nil {
		log.Fatal("Failed to read npy file", err)
	}

	for i := 0; i < npyDim; i++ {
		seedList[i].value = npyDat[i]
	}

	var acc float64
	var cnt int

	for i := 0; i < npyDim; i++ {
		acc += npyDat[i]
		cnt += 1
	}

	avg := acc / float64(cnt)
	scaleFactor := 7386 / avg

	for _, seed := range seedList {
		for xD := -1; xD < 2; xD++ {
			for yD := -1; yD < 2; yD++ {
				for zD := -1; zD < 2; zD++ {
					dist := xD*xD + yD*yD + zD*zD
					share := shareMap[dist]

					xPos := seed.x + xD
					yPos := seed.y + yD
					zPos := seed.z + zD

					vox := &fineMap[xPos][yPos][zPos]
					if vox.voxType == 1 {
						vox.value += (seed.value * scaleFactor * float64(share))
						vox.denom += share
					}
				}
			}
		}
	}

	averaging(&fineMap)

	newImg := nifti.NewImg(91, 109, 91, 1)
	var header nifti.Nifti1Header
	header.LoadHeader("template.nii")

	newImg.SetNewHeader(header)
	newImg.SetHeaderDim2(91, 109, 91, 1)

	for xPos := 0; xPos < 91; xPos++ {
		for yPos := 0; yPos < 109; yPos++ {
			for zPos := 0; zPos < 91; zPos++ {
				if fineMap[xPos][yPos][zPos].voxType == 2 {
					fineMap[xPos][yPos][zPos].value = 100
				}

				val := fineMap[xPos][yPos][zPos].value
				newImg.SetAt(uint32(xPos), uint32(yPos), uint32(zPos), 0, float32(val))
			}
		}
	}

	fmt.Println(newImg.GetHeader())

	newImg.Save("u.nii")
}

func avg0(fineMap *[91][109][91]Voxel, order <-chan int, wg *sync.WaitGroup) {
	for {
		xPos, ok := <-order
		if ok {
			for yPos := 0; yPos < 109; yPos++ {
				for zPos := 0; zPos < 91; zPos++ {
					vox := &fineMap[xPos][yPos][zPos]

					if vox.denom != 0 {
						val := vox.value
						vox.value = val / float64(vox.denom)
					}
				}
			}

			wg.Done()
		} else {
			break
		}
	}
}

func avg1(fineMap *[91][109][91]Voxel, order <-chan int, wg *sync.WaitGroup) {
	for {
		xPos, ok := <-order
		if ok {
			for yPos := 0; yPos < 109; yPos++ {
				for zPos := 0; zPos < 91; zPos++ {
					vox := &fineMap[xPos][yPos][zPos]

					if vox.denom != 0 {
						val := vox.value
						vox.value = val / float64(8)
					}
				}
			}

			wg.Done()
		} else {
			break
		}
	}
}

func averaging(fineMap *[91][109][91]Voxel) {
	order := make(chan int, runtime.NumCPU())
	var wg sync.WaitGroup

	wg.Add(91)

	for i := 0; i < runtime.NumCPU(); i++ {
		go avg0(fineMap, order, &wg)
	}

	for xPos := 0; xPos < 91; xPos++ {
		order <- xPos
	}

	wg.Wait()
	close(order)
}
