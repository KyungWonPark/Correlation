package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/KyungWonPark/Correlation/internal/cluster"
	"github.com/KyungWonPark/Correlation/internal/io"
)

func main() {
	fmt.Println("Clustering test...")

	thrs := []string{
		"0.000000", "0.050000", "0.100000", "0.150000", "0.200000", "0.250000", "0.300000", "0.350000", "0.400000", "0.450000",
		"0.500000", "0.550000", "0.600000", "0.650000", "0.700000", "0.750000",
		"0.800000", "0.810000", "0.820000", "0.830000", "0.840000", "0.850000", "0.860000", "0.870000", "0.880000", "0.890000",
		"0.900000", "0.910000", "0.920000", "0.930000", "0.940000", "0.950000", "0.960000", "0.970000", "0.980000", "0.990000",
	}

	stop := false
	reader := bufio.NewReader(os.Stdin)

	for !stop {
		txt, _ := reader.ReadString('\n')
		txt = strings.TrimSuffix(txt, "\n")

		if txt == "q" || txt == "Q" {
			break
		}

		n, err := strconv.Atoi(txt)
		if err != nil {
			fmt.Printf("Error! thats not a number")
		}
		fmt.Printf("---- ---- ---- ---- ---- ---- ---- ----\n")

		eigVal := io.NpytoMat64("/home/iksoochang2/kw-park/Result/gamma-0.001/eigVal-thr-" + thrs[n] + "-gamma-0.001000.npy")
		m := cluster.GetMultiplicity(eigVal, 0.001)
		fmt.Printf("Index: %d | Multiplicity: %d\n", n, m)

		fmt.Printf("---- ---- ---- ---- ---- ---- ---- ----\n")
	}

	return
}
