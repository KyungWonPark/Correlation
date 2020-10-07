package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"unsafe"

	"github.com/ghetzel/shmtool/shm"
)

func main() {

	shMem0, err := shm.Create(13362 * 13362 * 8)
	if err != nil {
		log.Fatal("Failed to create shared memory region")
	}

	pBase0, err := shMem0.Attach()
	if err != nil {
		log.Fatal("Failed to connect to shared memory region")
	}

	shMem1, err := shm.Create(13362 * 1 * 8)
	if err != nil {
		log.Fatal("Failed to create shared memory region")
	}

	pBase1, err := shMem1.Attach()
	if err != nil {
		log.Fatal("Failed to connect to shared memory region")
	}

	// Init
	for i := 0; i < 13362; i++ {
		for j := 0; j < 13362; j++ {
			index := uintptr(i*13362 + j)
			stride := uintptr(unsafe.Sizeof(float64(0)))

			addr := (*float64)(unsafe.Pointer(uintptr(pBase0) + index*stride))
			*addr = float64(i * j)
		}
	}

	for i := 0; i < 13362; i++ {
		index := uintptr(i)
		stride := uintptr(unsafe.Sizeof(float64(0)))

		addr := (*float64)(unsafe.Pointer(uintptr(pBase1) + index*stride))
		*addr = float64(i * 2)
	}

	fmt.Println("Call MAGMA Manually: ")
	fmt.Printf("./files/magma 13362 %d %d\n", shMem0.Id, shMem1.Id)
	fmt.Println("Press ENTER key to continue...")

	reader := bufio.NewReader(os.Stdin)
	reader.ReadString('\n')

	shMem0.Detach(pBase0)
	shMem0.Destroy()

	shMem1.Detach(pBase1)
	shMem1.Destroy()

	return
}
