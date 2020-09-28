package main

/*
#include "csource/sum.c"
*/
import "C"

import (
	"errors"
	"fmt"
	"log"
)

func main() {
	//Call to void function without params

	//Call to int function with two params
	res, err := makeSum(5, 4)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Sum of 5 + 4 is %d\n", res)
}

func makeSum(a, b int) (int, error) {
	//Convert Go ints to C ints
	sum, err := C.sum(C.int(a), C.int(b))
	if err != nil {
		return 0, errors.New("error calling Sum function: " + err.Error())
	}

	//Convert C.int result to Go int
	res := int(sum)

	return res, nil
}
