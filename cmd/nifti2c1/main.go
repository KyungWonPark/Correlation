package main

import (
	"fmt"

	"github.com/KyungWonPark/nifti"
)

func main() {
	// Open nitfi image
	var img nifti.Nifti1Image
	filePath := "bin/165941_rfMRI_REST1_LR-smoothed.nii"
	img.LoadImage(filePath, true)

	fmt.Println(img.GetDims())
	return
}
