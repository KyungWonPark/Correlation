package main

import "fmt"

// #cgo CFLAGS: -I.
// #cgo LDFLAGS: -L. -lgb
// #include <toto.h>
import "C"

func main() {
	fmt.Printf("Invoking C library...\n")
	fmt.Println("Done: ", C.x(11))
	return
}
