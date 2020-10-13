package anal

import (
	"log"
	"math"
	"sort"

	"github.com/gonum/matrix/mat64"
)

// Cluster represents cluster configuration
type Cluster [][]int

type tempCluster struct {
	members    map[float64][]int
	cntMembers map[float64]int
}

// GetNonZeroSmallestEigVal returns... as its name says
func GetNonZeroSmallestEigVal(eigVal *mat64.Dense) (float64, int) {
	n, _ := eigVal.Dims()

	if n < 2 {
		log.Fatal("[GetNonZeroSmallestEigVal] n is smaller than 2.")
	}

	type eig struct {
		idx   int
		value float64
	}

	temp := make([]eig, n)

	for i := 0; i < n; i++ {
		temp[i].idx = i
		temp[i].value = math.Abs(eigVal.At(i, 0))
	}

	sort.Slice(temp, func(i int, j int) bool {
		return temp[i].value < temp[j].value
	})

	return temp[1].value, temp[1].idx
}

// GetCluster creates cluster from eigenvalue and eigenvectors
func GetCluster(eigVal *mat64.Dense, eigVec *mat64.Dense) Cluster {
	_, idxNonZeroSmallestEigVal := GetNonZeroSmallestEigVal(eigVal)

	var c tempCluster

	rows, _ := eigVec.Dims()
	for i := 0; i < rows; i++ {
		val := eigVec.At(i, idxNonZeroSmallestEigVal)
		c.members[val] = append(c.members[val], i)
		c.cntMembers[val] = c.cntMembers[val] + 1
	}

	var realC Cluster

	for k := range c.members {
		if c.cntMembers[k] >= 2 {
			members := []int{}
			for _, v := range c.members[k] {
				members = append(members, v)
			}

			sort.Ints(members)

			realC = append(realC, members)
		}
	}

	return realC
}

// ANC calculates Average Number of Clusters

// WCC calculates Within Cluster Correlation
